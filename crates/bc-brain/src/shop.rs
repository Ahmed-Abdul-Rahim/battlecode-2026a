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
/// camp kills destroy carried gems; Monster next (10% more camp damage).
pub fn build_order(kind: ChampionKind) -> Vec<(Family, Stat)> {
    build_order_profile(kind, 0)
}

/// Build order by profile (θ_match knob BUILD_PROFILE; chosen per map from tournaments):
/// 0 default (Jungle(Health) first), 1 Monster first, 2 all primary stat,
/// 3 Power/Bruiser primary (as strong ladder teams buy), 4 Power/Jungle/Monster primary,
/// then default order after: 5 Power ×2, 6 Treasure, 7 Durability(Health), 8 Home, 9 Bruiser;
/// 10 RedDragon (Monster, Power, all Strength), 11 every passive on the primary stat.
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
        _ => {}
    }
    default_order(kind)
}

fn default_order(kind: ChampionKind) -> Vec<(Family, Stat)> {
    let primary = primary_stat(kind);
    let mut order = vec![(Family::Jungle, Stat::Health)];
    if kind == ChampionKind::VesperVale {
        // +20% gold stacks with her passive for +40%.
        order.push((Family::Treasure, Stat::Strength));
    }
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

pub fn affordable(gold: u32) -> u32 {
    gold / SHARD_COST
}

#[cfg(test)]
mod tests {
    use super::*;

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
