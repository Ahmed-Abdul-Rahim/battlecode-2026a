// Fuel benchmark bot: measures the fuel cost of SDK calls and of guest-side
// algorithms so docs/fuel-budget.md and the feature-selection cost term use
// real numbers. Prints `FUEL <name> <fuel>` lines to stdout.
use cambc::{Bitboard, DistanceField, Result, Team, Vec2};
use std::collections::VecDeque;

fn fuel() -> u64 {
    cambc::fuel_remaining().unwrap_or(0)
}

macro_rules! bench {
    ($name:expr, $body:expr) => {{
        let before = fuel();
        let result = $body;
        let after = fuel();
        println!("FUEL {} {}", $name, before.saturating_sub(after));
        result
    }};
}

fn draft() -> Result<()> {
    let chosen = cambc::draft::current_draft(Team::Ally)?;
    for i in 0..11 {
        let pick = cambc::ChampionDraft::try_from(i)?;
        if !chosen.contains(&pick) {
            return cambc::draft::pick(pick);
        }
    }
    Ok(())
}

fn rust_bfs(blocked: &[bool], width: usize, height: usize, start: (usize, usize)) -> usize {
    let mut dist = vec![u16::MAX; width * height];
    let mut queue = VecDeque::with_capacity(width * height);
    dist[start.1 * width + start.0] = 0;
    queue.push_back(start);
    let mut reached = 0;
    while let Some((x, y)) = queue.pop_front() {
        reached += 1;
        let d = dist[y * width + x];
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let i = ny as usize * width + nx as usize;
                if !blocked[i] && dist[i] == u16::MAX {
                    dist[i] = d + 1;
                    queue.push_back((nx as usize, ny as usize));
                }
            }
        }
    }
    reached
}

fn int8_layer(weights: &[i8], input: &[i8], out: &mut [i8]) {
    let n_in = input.len();
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &weights[o * n_in..(o + 1) * n_in];
        let mut acc: i32 = 0;
        for (w, x) in row.iter().zip(input) {
            acc += i32::from(*w) * i32::from(*x);
        }
        *slot = (acc.max(0) >> 7).min(127) as i8;
    }
}

fn unpack_bits(bytes: &[u8], n: usize) -> Vec<bool> {
    (0..n).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()
}

fn run_benches() -> Result<()> {
    let me = cambc::self_champion()?.expect("gameplay");
    let origin = me.position()?;
    let width = cambc::map::width()? as usize;
    let height = cambc::map::height()? as usize;
    println!("MAP {width} {height}");

    bench!("noop", ());
    bench!("loop_1000_adds", {
        let mut s = 0u32;
        for i in 0..1000u32 {
            s = s.wrapping_add(i ^ 0x55);
        }
        std::hint::black_box(s)
    });
    bench!("println_short", println!("x"));

    let walls = bench!("map_walls_borrow", cambc::map::walls()?);
    let mut blocked = bench!("bitboard_clone", walls.clone());
    bench!("bitboard_union_assign", blocked.union_assign(cambc::map::holes()?)?);
    blocked.union_assign(cambc::map::altars()?)?;
    bench!("bitboard_new", Bitboard::new()?);
    bench!("vision_champion_tiles", cambc::vision::champion_tiles(None)?);
    bench!("vision_camp_tiles", cambc::vision::camp_tiles()?);
    bench!("vision_visible_tiles", cambc::vision::visible_tiles()?);
    let champs = bench!("champions_all", cambc::champions(None)?);
    bench!("vision_champions_enemy", cambc::vision::champions(Some(Team::Enemy))?);
    let camps = bench!("vision_camps", cambc::vision::camps()?);
    let groups = bench!("map_camp_groups", cambc::map::camp_groups()?);
    println!("COUNT champions {} camps {} groups {}", champs.len(), camps.len(), groups.len());

    bench!("me_position", me.position()?);
    bench!("me_health", me.health()?);
    bench!("me_gold_gems", (me.gold_count()?, me.gem_count()?));
    bench!("me_item_count", me.item_count()?);
    bench!("me_can_move_attack_cast", (me.can_move()?, me.can_attack()?, me.can_cast()?));
    bench!("me_ability0_cooldown", me.abilities()[0].cooldown()?);
    bench!("me_attack_tiles", me.attack_tiles()?);
    bench!("round_number", cambc::round_number()?);
    bench!("global_submitted_gems", cambc::global_submitted_gems(Team::Ally)?);
    bench!("shared_memory_read8", {
        let mut b = [0u8; 8];
        cambc::shared_memory().read(0, &mut b)?;
        b
    });

    let mut roots = Bitboard::new()?;
    for p in cambc::map::altars()?.positions()?.into_iter().take(1) {
        roots.draw_chebyshev(p, 1.0)?;
    }
    roots.intersect_assign(&blocked.complement()?)?;
    let mut field = bench!("distancefield_new", DistanceField::new()?);
    let mut visited = Bitboard::new()?;
    bench!(
        "distancefield_compute_full",
        field.compute(&blocked, &roots, &mut visited, DistanceField::UNREACHABLE)?
    );
    bench!("distancefield_next_moves", field.next_moves_from(origin)?);
    let mut here = Bitboard::new()?;
    here.set(origin, true)?;
    let mut local = DistanceField::new()?;
    bench!(
        "distancefield_compute_to_r12",
        local.compute_to(&blocked, &roots, &here, &mut visited, 12)?
    );
    bench!("bitboard_reachable_r1", blocked.reachable_tiles(origin, 1)?);
    bench!("bitboard_reachable_r8", blocked.reachable_tiles(origin, 8)?);
    let mut opaque = walls.clone();
    opaque.union_assign(cambc::map::bushes()?)?;
    bench!("bitboard_shadow_cast_r8", opaque.shadow_cast(origin, 8)?);
    bench!("bitboard_positions_altars", cambc::map::altars()?.positions()?);
    bench!("bitboard_count", blocked.count()?);
    bench!("distancefield_export_linear", field.export_linear_buffer()?);
    Ok(())
}

fn run_benches_b() -> Result<()> {
    let me = cambc::self_champion()?.expect("gameplay");
    let origin = me.position()?;
    let width = cambc::map::width()? as usize;
    let height = cambc::map::height()? as usize;
    let mut blocked = cambc::map::walls()?.clone();
    blocked.union_assign(cambc::map::holes()?)?;
    let mut roots = Bitboard::new()?;
    roots.set(origin, true)?;
    let mut field = DistanceField::new()?;
    let mut visited = Bitboard::new()?;
    field.compute(&blocked, &roots, &mut visited, DistanceField::UNREACHABLE)?;
    bench!("distancefield_get_x100", {
        let mut s = 0u32;
        for i in 0..100 {
            s += u32::from(field.get(Vec2::new(i % width as i32, i / width as i32))?);
        }
        s
    });
    bench!("bitboard_is_set_x100", {
        let mut s = 0u32;
        for i in 0..100 {
            s += u32::from(blocked.is_set(Vec2::new(i % width as i32, i / width as i32))?);
        }
        s
    });
    let bits = bench!("bitboard_export_linear", blocked.export_linear_buffer()?);
    let grid = bench!("unpack_bits_full_map", unpack_bits(&bits, width * height));
    // Windowed BFS: 15x15 around origin, the size planned for guest-side weighted search.
    let x0 = (origin.x as usize).saturating_sub(7).min(width - 15);
    let y0 = (origin.y as usize).saturating_sub(7).min(height - 15);
    let window: Vec<bool> = (0..15 * 15)
        .map(|i| grid[(y0 + i / 15) * width + x0 + i % 15])
        .collect();
    let start = (origin.x as usize - x0, origin.y as usize - y0);
    let reached = bench!("rust_bfs_15x15", rust_bfs(&window, 15, 15, start));
    println!("COUNT bfs_reached {reached}");
    Ok(())
}

fn run_benches_c() -> Result<()> {
    let width = cambc::map::width()? as usize;
    let height = cambc::map::height()? as usize;

    let mut seed = 0x1234_5678u32;
    let weights: Vec<i8> = (0..64 * 32)
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 24) as i8
        })
        .collect();
    let input: Vec<i8> = (0..64).map(|i| (i * 3 % 127) as i8).collect();
    let mut out = [0i8; 32];
    bench!("int8_layer_64x32", int8_layer(&weights, &input, &mut out));
    let w2: Vec<i8> = weights[..32 * 16].to_vec();
    let mut out2 = [0i8; 16];
    bench!("int8_layer_32x16", int8_layer(&w2, &out, &mut out2));
    std::hint::black_box(out2);

    // Bit order probe: which bit does (x, y) land in?
    let mut probe = Bitboard::new()?;
    probe.set(Vec2::new(1, 0), true)?;
    probe.set(Vec2::new(0, 1), true)?;
    let pb = probe.export_linear_buffer()?;
    // (1,0) -> index 1; (0,1) -> index width if row-major with x fastest.
    let row1 = width / 8;
    println!(
        "BITORDER byte0={:#010b} byte_row1[{row1}]={:#010b} width={width} height={height} (lsb-first if byte0==0b10)",
        pb[0], pb[row1]
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut turn = 0u32;
    loop {
        if cambc::is_draft() {
            if let Err(e) = draft() {
                println!("draft error: {e}");
            }
        } else {
            turn += 1;
            if turn == 1 {
                println!("FIRST_TURN_REMAINING {}", fuel());
            }
            let result = match turn {
                3 => Some(run_benches()),
                5 => Some(run_benches_b()),
                7 => Some(run_benches_c()),
                _ => None,
            };
            if let Some(result) = result {
                if let Err(e) = result {
                    println!("bench error: {e}");
                }
                println!("END_REMAINING turn {turn} {}", fuel());
            }
        }
        cambc::end_turn();
    }
}
