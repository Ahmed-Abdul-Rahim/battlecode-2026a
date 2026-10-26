//! Build orders: a list of emblems to complete, in order. Each entry is five
//! matching shards (Power gives a sixth shard's worth of stats on completion).

use bc_core::champion::ChampionKind;
use bc_core::ledger::{item_id, Family, Ledger, Stat, SHARD_COST};

/// The stat a champion scales with most.
pub fn primary_stat(kind: ChampionKind) -> Stat {
    use ChampionKind as C;
    match kind {
        // Ryth converts bonus health into strength; health also heals on spawn.
        C::Ryth => Stat::Health,
        C::Stellar | C::Prism | C::Cassi | C::Tavi | C::Willow => Stat::Focus,
        _ => Stat::Strength,
    }
}

/// Emblems to complete, in order. Jungle first (30% less camp damage) because
/// camp kills destroy carried gems; Monster next (15% more camp damage since engine 2.0.0).
pub fn build_order(kind: ChampionKind) -> Vec<(Family, Stat)> {
    build_order_profile(kind, 0)
}

/// Build order by profile (θ_match knob BUILD_PROFILE; chosen per map from tournaments):
/// 0 default (Jungle(Health) first), 1 Monster first, 2 all primary stat,
/// 3 Power/Bruiser primary (as strong ladder teams buy), 4 Power/Jungle/Monster primary,
/// then default order after: 5 Power ×2, 6 Treasure, 7 Durability(Health), 8 Home, 9 Bruiser;
/// 10 RedDragon (Monster, Power, all Strength), 11 every passive on the primary stat,
/// 12 Tavi field build (every passive on Strength, then Power), 13 best-bot Power build;
/// engine 2.0.0 families: 14 Skirmish/Hunter fighter, 15 Capture/Prospector farmer,
/// 16 Greed (25 shards for 35 shards of stats), 17 Collectors (one emblem per family).
pub fn build_order_profile(kind: ChampionKind, profile: u8) -> Vec<(Family, Stat)> {
    let primary = primary_stat(kind);
    let (p, j, m, b) = (
        (Family::Power, primary),
        (Family::Jungle, primary),
        (Family::Monster, primary),
        (Family::Bruiser, primary),
    );
    match profile {
        1 | 5..=9 => {
            let first: &[(Family, Stat)] = match profile {
                1 => &[m],
                5 => &[p, p],
                6 => &[(Family::Treasure, primary)],
                7 => &[(Family::Durability, Stat::Health)],
                8 => &[(Family::Home, Stat::Health)],
                _ => &[b],
            };
            let mut order = first.to_vec();
            order.extend(default_order(kind));
            return order;
        }
        2 => return [j, m].into_iter().chain([p; 9]).collect(),
        3 => return [p, b, p, p, j].into_iter().chain([p; 6]).collect(),
        4 => return [p, j, m].into_iter().chain([p; 8]).collect(),
        // RedDragon's build (ladder): Monster then Power, all Strength, every champion.
        10 => {
            let (ms, ps) = ((Family::Monster, Stat::Strength), (Family::Power, Stat::Strength));
            return [ms].into_iter().chain([ps; 10]).collect();
        }
        // Strong ladder teams (runs/ranked_v5): every passive once, all on the primary stat
        // (the passive doesn't depend on the stat), Monster first, then stack Power.
        11 => {
            let (h, d) = ((Family::Home, primary), (Family::Durability, primary));
            let mut order = vec![m];
            if kind == ChampionKind::VesperVale {
                order.push((Family::Treasure, primary));
            }
            order.extend([j, h, d]);
            order.extend([p; 7]);
            return order;
        }
        // Best ladder bot: Power on the damage stat from the first emblem (Power counts as six
        // shards), one Durability and one Home (Health) along the way. Ryth on Strength: its
        // passive turns a Health shard into only +1 Strength, a Strength shard gives +5.
        13 => {
            let damage = if kind == ChampionKind::Ryth { Stat::Strength } else { primary };
            let pw = (Family::Power, damage);
            let mut order = vec![pw; 4];
            order.push((Family::Durability, Stat::Health));
            order.extend([pw; 4]);
            order.push((Family::Home, Stat::Health));
            order.extend([pw; 6]);
            return order;
        }
        // Tavi (ladder meta): every passive on Strength (basic attack + reset), then Power.
        12 => {
            let s = |f| (f, Stat::Strength);
            let mut order: Vec<(Family, Stat)> =
                [Family::Monster, Family::Jungle, Family::Home, Family::Durability].map(s).to_vec();
            order.extend([s(Family::Power); 8]);
            return order;
        }
        // Fighter: +5% damage to champions and +50 gold per kill, then stack Power.
        14 => {
            let mut order = vec![(Family::Skirmish, primary), (Family::Hunter, primary), m];
            order.extend([p; 8]);
            return order;
        }
        // Farmer: captures in 3 turns instead of 5, +25% cluster gold and gems, Monster's
        // +15% camp damage, then Power.
        15 => {
            let mut order =
                vec![(Family::Capture, primary), (Family::Prospector, primary), m, j];
            order.extend([p; 7]);
            return order;
        }
        // Greed: one Monster for early camps, then Greed emblems (35 shards of stats per
        // 25 shards bought, 1.4x vs Power's 1.2x, but nothing until the 25th shard).
        16 => {
            let g = (Family::Greed, primary);
            return vec![m, g, g, g];
        }
        // Collectors: +1% damage per unique family, so one emblem of each useful family
        // (Collectors counts itself), then Power.
        17 => {
            let mut order = vec![
                (Family::Collectors, primary),
                m,
                j,
                (Family::Skirmish, primary),
                (Family::Capture, primary),
                (Family::Durability, Stat::Health),
                (Family::Home, Stat::Health),
                (Family::Tribute, primary),
            ];
            order.extend([p; 4]);
            return order;
        }
        _ => {}
    }
    default_order(kind)
}

fn default_order(kind: ChampionKind) -> Vec<(Family, Stat)> {
    let primary = primary_stat(kind);
    // (Vesper Vale used to take a Treasure emblem here for +20% gold; engine 2.0.0 cut it to
    // +10%, less than the Power emblem it delays. Profiles 6 and 11 still offer it.)
    let order = vec![(Family::Jungle, Stat::Health)];
    let mut order = order;
    order.push((Family::Monster, primary));
    order.push((Family::Power, primary));
    order.push((Family::Durability, Stat::Health));
    order.extend([(Family::Power, primary); 6]);
    order
}

/// Next shard to buy given what we own, or `None` when the order is complete.
pub fn next_shard(order: &[(Family, Stat)], ledger: &Ledger) -> Option<u8> {
    let mut needed: Vec<((Family, Stat), u8)> = Vec::new();
    for &entry in order {
        match needed.iter_mut().find(|(e, _)| *e == entry) {
            Some((_, n)) => *n += 1,
            None => needed.push((entry, 1)),
        }
        let (family, stat) = entry;
        let want = needed.iter().find(|(e, _)| *e == entry).map_or(1, |(_, n)| *n);
        let have = ledger.items[item_id(family, stat, true) as usize];
        if have < want {
            return Some(item_id(family, stat, false));
        }
    }
    None
}

/// Like [`next_shard`], but once the order is complete keep adding its last emblem (usually
/// Power on the primary stat) instead of hoarding gold. `None` only for an empty order.
pub fn next_shard_or_repeat(order: &[(Family, Stat)], ledger: &Ledger) -> Option<u8> {
    next_shard(order, ledger).or_else(|| order.last().map(|&(f, s)| item_id(f, s, false)))
}

pub fn affordable(gold: u32) -> u32 {
    gold / SHARD_COST
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine2_profiles_use_new_families() {
        for profile in 14..=17 {
            let order = build_order_profile(ChampionKind::Varg, profile);
            assert!(order.iter().any(|&(f, _)| f as u8 >= Family::Tribute as u8), "{profile}");
        }
        // Greed needs 25 shards per emblem: the 6th Greed shard is still Greed.
        let order = build_order_profile(ChampionKind::Varg, 16);
        let mut l = Ledger::new(ChampionKind::Varg);
        l.items[item_id(Family::Monster, Stat::Strength, true) as usize] = 1;
        l.items[item_id(Family::Greed, Stat::Strength, false) as usize] = 7;
        assert_eq!(next_shard(&order, &l), Some(item_id(Family::Greed, Stat::Strength, false)));
    }

    #[test]
    fn complete_order_keeps_buying_its_last_emblem() {
        let order = build_order(ChampionKind::Varg);
        let mut l = Ledger::new(ChampionKind::Varg);
        for &(f, s) in &order {
            l.items[item_id(f, s, true) as usize] += 1;
        }
        assert_eq!(next_shard(&order, &l), None);
        let (f, s) = *order.last().unwrap();
        assert_eq!(next_shard_or_repeat(&order, &l), Some(item_id(f, s, false)));
        assert_eq!(next_shard_or_repeat(&[], &l), None);
    }

    #[test]
    fn order_starts_with_jungle_and_advances_with_emblems() {
        let order = build_order(ChampionKind::Varg);
        let mut l = Ledger::new(ChampionKind::Varg);
        assert_eq!(next_shard(&order, &l), Some(item_id(Family::Jungle, Stat::Health, false)));
        l.items[item_id(Family::Jungle, Stat::Health, true) as usize] = 1;
        assert_eq!(next_shard(&order, &l), Some(item_id(Family::Monster, Stat::Strength, false)));
    }

    #[test]
    fn repeated_power_entries_need_repeated_emblems() {
        let order = build_order(ChampionKind::Stellar);
        let mut l = Ledger::new(ChampionKind::Stellar);
        for (f, s) in [
            (Family::Jungle, Stat::Health),
            (Family::Monster, Stat::Focus),
            (Family::Durability, Stat::Health),
        ] {
            l.items[item_id(f, s, true) as usize] = 1;
        }
        l.items[item_id(Family::Power, Stat::Focus, true) as usize] = 3;
        assert_eq!(next_shard(&order, &l), Some(item_id(Family::Power, Stat::Focus, false)));
        l.items[item_id(Family::Power, Stat::Focus, true) as usize] = 7;
        assert_eq!(next_shard(&order, &l), None);
    }
}
