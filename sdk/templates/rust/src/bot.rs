// Champion decisions. main.rs dispatches here after the draft.
use cambc::{
    Bitboard, CampGroup, Champion,
    DistanceField, ItemType as Item, Result, Team, Unit, Vec2,
};
use std::ops::Deref;

fn me() -> Result<&'static Champion> {
    Ok(cambc::self_champion()?.expect("play starts after the draft"))
}

fn distance(a: Vec2, b: Vec2) -> i32 {
    a.chebyshev_distance(b) as i32
}

fn tiles_around(position: Vec2, radius: i32) -> Result<Bitboard> {
    let mut tiles = Bitboard::new()?;
    tiles.draw_chebyshev(position, radius as f32)?;
    Ok(tiles)
}

fn blocked_tiles() -> Result<Bitboard> {
    // Combine terrain and visible occupants using native bitboard operations.
    let mut blocked = cambc::map::walls()?.clone();
    for mask in [cambc::map::holes()?.clone(), cambc::map::altars()?.clone(),
        cambc::map::spawn_region(Team::Enemy)?.clone(), cambc::vision::champion_tiles(None)?,
        cambc::vision::camp_tiles()?, cambc::vision::teleport_tiles()?] {
        blocked.union_assign(&mask)?;
    }
    blocked.set(me()?.position()?, false)?;
    Ok(blocked)
}

fn move_towards_closest(mut goals: Bitboard) -> Result<bool> {
    let me = me()?;
    let origin = me.position()?;
    if !me.can_move()? || goals.is_set(origin)? { return Ok(false); }

    let blocked = blocked_tiles()?;
    goals.intersect_assign(&blocked.complement()?.union(&cambc::vision::camp_tiles()?)?)?;
    if goals.is_empty()? { return Ok(false); }
    if goals.count()? == 1 && !blocked.is_set(goals.positions()?[0])? {
        return me.move_towards(goals.positions()?[0]);
    }

    let mut field = DistanceField::new()?;
    field.compute_to(&blocked.intersect(&goals.complement()?)?, &goals, &tiles_around(origin, 0)?, &mut Bitboard::new()?, DistanceField::UNREACHABLE)?;

    let mut step = origin;
    for tile in blocked.reachable_tiles(origin, me.base_stats()?.base_move_speed as u16)?.positions()? {
        if field.get(tile)? < field.get(step)? { step = tile; }
    }
    if step == origin { Ok(false) } else { me.move_absolute(step) }
}

// Compute the attack mask once, rather than once per visible target.
fn attack_first_in_range<T: Deref<Target = Unit>>(targets: &[T]) -> Result<bool> {
    let me = me()?;
    if targets.is_empty() || !me.can_attack()? { return Ok(false); }
    let attack_tiles = me.attack_tiles()?;
    for target in targets {
        if target.is_visible()? && attack_tiles.is_set(target.position()?)? && target.health()? > 0.0 {
            me.attack(target.deref())?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn farm_camps(bot: &mut Bot) -> Result<()> {
    // Attack a camp already in range, otherwise move toward the nearest camp tiles.
    if attack_first_in_range(&cambc::vision::camps()?)? { return Ok(()); }

    // Cache fixed spawn locations; only visible respawn state changes each turn.
    if bot.camp_layout.is_none() {
        let mut tiles = Bitboard::new()?;
        let mut groups = Vec::new();
        for group in cambc::map::camp_groups()? {
            let spawns = group.spawn_tiles()?;
            tiles.union_assign(&spawns)?;
            groups.push((group, spawns));
        }
        bot.camp_layout = Some((tiles, groups));
    }
    let (spawns, groups) = bot.camp_layout.as_ref().unwrap();
    let mut goals = spawns.clone();
    let visible = cambc::vision::visible_tiles()?;
    for (group, tiles) in groups {
        if !tiles.intersect(&visible)?.is_empty()?
            && matches!(group.state()?, cambc::CampGroupState::Respawning(_)) {
            goals.intersect_assign(&tiles.complement()?)?;
        }
    }

    if move_towards_closest(goals)? {
        attack_first_in_range(&cambc::vision::camps()?)?;
    }
    Ok(())
}

// Per-champion state survives between turns. Pick one stat and keep buying it.
#[derive(Default)]
pub struct Bot {
    chosen_stat: Option<usize>,
    camp_layout: Option<(Bitboard, Vec<(CampGroup, Bitboard)>)>,
}

impl Bot {
    fn buy_items(&self) -> Result<()> {
        // Spend only on power shards of the randomly chosen stat.
        let me = me()?;
        let item = [Item::PowerShardOfStrength, Item::PowerShardOfFocus, Item::PowerShardOfHealth][self.chosen_stat.unwrap()];
        let count = me.gold_count()? / item.cost()?;
        if count > 0 && me.can_buy(item, count)? { me.buy_count(item, count)?; }
        Ok(())
    }

    pub fn play_turn(&mut self) -> Result<()> {
        if self.chosen_stat.is_none() {
            self.chosen_stat = Some(crate::random_number(3) as usize);
            println!("draft: {}", String::from_utf8_lossy(&cambc::draft_memory().bytes()[..11]));
        }
        // Read before writing: little-endian u32 at shared-memory byte offset zero.
        let mut bytes = [0; 4];
        cambc::shared_memory().read(0, &mut bytes)?;
        println!("shared round: {}", u32::from_le_bytes(bytes));
        cambc::shared_memory().write(0, &cambc::round_number()?.to_le_bytes())?;

        // Priorities: shop and recall, attack enemies in range, bank gems, then farm.
        let me = me()?;
        if !me.is_alive()? || me.is_recalling()? || me.is_submitting_gems()? || me.is_teleporting()?
            || (!me.can_move()? && !me.can_attack()?) { return Ok(()); }

        let at_spawn = cambc::map::spawn_region(Team::Ally)?.is_set(me.position()?)?;
        if at_spawn { self.buy_items()?; }

        if !at_spawn && me.health()? < me.max_health()? * 0.35 {
            me.recall()?;
            return Ok(());
        }

        if attack_first_in_range(&cambc::vision::champions(Some(Team::Enemy))?)? { return Ok(()); }

        if me.gem_count()? > 0 {
            // Union all altar approach tiles: the field chooses the nearest reachable altar.
            let mut goals = Bitboard::new()?;
            for altar in cambc::map::altars()?.positions()? {
                if cambc::vision::has_altar_lock(altar)? { continue; }
                let range = cambc::constants().submit_gems_range;
                if distance(me.position()?, altar) <= range { me.submit_gems(altar)?; return Ok(()); }
                goals.union_assign(&tiles_around(altar, range)?)?;
            }
            move_towards_closest(goals)?;
            return Ok(());
        }

        farm_camps(self)?;

        cambc::self_champion()?.expect("gameplay champion").set_priority(1 + crate::random_number(100) as i32)
    }
}

