//! SDK-free core types shared by every logic crate.
//!
//! Nothing here calls the `cambc` SDK, so all of it is unit-testable on the
//! host. `bots/main` converts SDK values into these types once per turn.

pub mod belief;
pub mod champion;
pub mod combat;
pub mod crc;
pub mod fuel;
pub mod geom;
pub mod grid;
pub mod ledger;
pub mod memory;

pub use geom::Pos;
pub use grid::{BitGrid, Terrain};
