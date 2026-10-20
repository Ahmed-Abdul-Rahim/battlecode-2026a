//! Per-champion combat state machine with short history (knob COMBAT_FSM).
//!
//! The rule layers in the bot decide each turn from scratch, so a champion near the
//! fight/flee line flips every turn, goes back to farming the moment an enemy steps out of
//! view, and walks away from allies who asked for help. This module keeps a little state
//! between turns:
//!
//! * a state ([`State`]) with minimum dwell times: ECONOMY, ALERT (an enemy is around: no
//!   channels, no camp waking), ENGAGE, DISENGAGE, RECOVER, ASSIST (join an ally's fight)
//!   and ESCORT (cover an ally retreating with loot);
//! * 2-bit saturating counters ([`Ctr2`], as in a branch predictor): the fight decision only
//!   flips after two turns of evidence;
//! * a shift register of recent exchange results and a counter per enemy champion kind
//!   ("fights against this kind keep going badly").
//!
//! [`Fsm::step`] turns this turn's observations into a [`Directive`] the bot maps onto its
//! intent, plus the help [`Call`] to broadcast in the team channel (memory.rs). Pure and
//! SDK-free: everything is decided from [`Inputs`]. Cost: a few hundred instructions.

use crate::geom::Pos;

/// 2-bit saturating counter: 0 strong no, 1 weak no, 2 weak yes, 3 strong yes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ctr2(u8);

impl Default for Ctr2 {
    fn default() -> Self {
        Self(1)
    }
}

impl Ctr2 {
    pub fn update(&mut self, yes: bool) {
        self.0 = if yes { (self.0 + 1).min(3) } else { self.0.saturating_sub(1) };
    }

    #[must_use]
    pub fn yes(self) -> bool {
        self.0 >= 2
    }

    #[must_use]
    pub fn strong_yes(self) -> bool {
        self.0 == 3
    }

    #[must_use]
    pub fn strong_no(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub fn value(self) -> u8 {
        self.0
    }
}

/// Champion kinds (`ChampionKind as usize`); one counter each.
pub const KINDS: usize = 12;

/// What happened in our recent fights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct History {
    /// Last exchanges, newest in bit 0: 1 = we dealt at least as much damage as we took.
    pub results: u8,
    /// How many exchanges `results` holds (saturates at 8).
    pub len: u8,
    /// Per enemy kind: fights against it go well (yes) or badly (no).
    pub vs_kind: [Ctr2; KINDS],
}

impl History {
    /// Record one turn of fighting: damage we dealt to and took from enemy champions.
    pub fn record(&mut self, kind: Option<usize>, dealt: f32, taken: f32) {
        if dealt <= 0.0 && taken <= 0.0 {
            return;
        }
        let won = dealt >= taken;
        self.results = self.results << 1 | u8::from(won);
        self.len = (self.len + 1).min(8);
        if let Some(c) = kind.and_then(|k| self.vs_kind.get_mut(k)) {
            c.update(won);
        }
    }

    /// Lost exchanges among the last `n` recorded.
    #[must_use]
    pub fn recent_losses(&self, n: u8) -> u32 {
        let n = n.min(self.len);
        if n == 0 {
            return 0;
        }
        let mask = if n >= 8 { 0xFF } else { (1u8 << n) - 1 };
        n as u32 - (self.results & mask).count_ones()
    }

    /// Our record against this kind says stay away.
    #[must_use]
    pub fn bad_against(&self, kind: usize) -> bool {
        self.vs_kind.get(kind).is_some_and(|c| c.strong_no())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum State {
    #[default]
    Economy,
    Alert,
    Engage,
    Disengage,
    Recover,
    Assist,
    Escort,
}

impl State {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Economy => "economy",
            Self::Alert => "alert",
            Self::Engage => "engage",
            Self::Disengage => "disengage",
            Self::Recover => "recover",
            Self::Assist => "assist",
            Self::Escort => "escort",
        }
    }
}

/// What a help call asks for (2 bits in the team channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallIntent {
    /// I'm fighting and can win with help: come and attack.
    Fight = 0,
    /// I'm retreating (usually with loot): meet me on my way home, stop the chaser.
    Escape = 1,
    /// The enemy is almost dead: come take the kill if you're close.
    Finish = 2,
    /// Lost or dead: don't come (anyone on the way turns back).
    Abort = 3,
}

impl CallIntent {
    #[must_use]
    pub fn from_bits(b: u8) -> Self {
        match b & 0b11 {
            0 => Self::Fight,
            1 => Self::Escape,
            2 => Self::Finish,
            _ => Self::Abort,
        }
    }
}

/// A help call to broadcast: where the caller is and how long it can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub pos: Pos,
    pub intent: CallIntent,
    /// 3 = plenty (10+ turns or winning), 2 = 5-9, 1 = 2-4, 0 = none.
    pub time_left: u8,
}

/// Turns of help a `time_left` bucket promises (lower edge of the bucket).
#[must_use]
pub fn time_left_turns(bucket: u8) -> i32 {
    match bucket {
        0 => 0,
        1 => 2,
        2 => 5,
        _ => 10,
    }
}

/// Bucket a survival estimate in turns.
#[must_use]
pub fn time_left_bucket(turns: f32) -> u8 {
    if turns >= 10.0 {
        3
    } else if turns >= 5.0 {
        2
    } else if turns >= 2.0 {
        1
    } else {
        0
    }
}

/// The nearest relevant enemy this turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnemyIn {
    pub pos: Pos,
    pub dist: i32,
    /// Attack range (it also moves one tile a turn).
    pub range: i32,
    pub kind: usize,
    pub hp_frac: f32,
    /// Our fight advantage against it (router::fight_advantage x learned bias); 1 = even.
    pub advantage: f32,
}

/// A teammate's help call as this champion sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallIn {
    pub pos: Pos,
    pub intent: CallIntent,
    pub time_left: u8,
    /// Rounds since it was written (0 = this round).
    pub age: u32,
    /// Turns we need to get there.
    pub my_eta: i32,
    /// Where to stand to cover an escaping caller (on its way home).
    pub escort_point: Pos,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inputs {
    pub round: u32,
    pub alive: bool,
    pub me: Pos,
    pub hp_frac: f32,
    /// Carrying enough gems that dying hands the enemy a big bag.
    pub big_bag: bool,
    pub enemy: Option<EnemyIn>,
    /// Turns until any seen or remembered enemy can hit us (0 = now); None = no threat.
    pub threat_eta: Option<i32>,
    pub hurt: bool,
    /// Turns we survive the current exchange (our hp / incoming dps).
    pub survive_turns: f32,
    /// Advantage needed to engage (1 + fight margin).
    pub fight_threshold: f32,
    pub call: Option<CallIn>,
    /// Turns a recall takes (5, or 3 with Home).
    pub recall_ticks: i32,
}

/// What the bot should do this turn (mapped onto its intent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Directive {
    /// Leave the intent as the other layers chose it.
    Keep,
    /// Keep farming, but start no channel (capture, tribute) and wake no camp.
    NoChannel,
    /// Attack the enemy at this position.
    Engage(Pos),
    /// Walk away towards home without recalling.
    Disengage,
    /// Go home: recall when safe.
    Recover,
    /// Join an ally's fight here.
    Assist(Pos),
    /// Stand on an escaping ally's path home.
    Escort(Pos),
}

/// Tuning (trainable through learned.rs X slots; see `Tuning::from_learned`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    /// Quiet turns (no enemy in sight or reach) before ALERT returns to ECONOMY.
    pub alert_turns: u32,
    /// Multiplier on the time a caller says it can hold when deciding to go help.
    pub assist_eta_scale: f32,
    /// Only champions above this health fraction answer calls.
    pub assist_min_hp: f32,
    /// Below this health fraction an engaged champion disengages unless it's winning clearly.
    pub retreat_hp: f32,
    /// Turn and fight instead of running when the enemy keeps hitting us and the fight is at
    /// least this even (running from a faster enemy only gives it free hits).
    pub last_stand_advantage: f32,
    /// ALERT starts when a remembered threat can reach us within this many turns.
    pub alert_eta: i32,
    /// After getting away, go home (recall) only below this health fraction or with a big
    /// bag; otherwise stay on alert nearby and keep farming. Above 1.0 = always go home.
    pub recover_below_hp: f32,
    /// Answer teammates' help calls at all.
    pub answer_calls: bool,
    /// Never travel more than this many turns to help.
    pub max_help_eta: i32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            alert_turns: 6,
            assist_eta_scale: 1.0,
            assist_min_hp: 0.5,
            retreat_hp: 0.35,
            last_stand_advantage: 0.8,
            alert_eta: 6,
            recover_below_hp: 1.01,
            answer_calls: true,
            max_help_eta: 99,
        }
    }
}

impl Tuning {
    /// From the two learned X slots: 0 (old packs) keeps the defaults.
    #[must_use]
    pub fn from_learned(alert: f32, assist: f32) -> Self {
        let mut t = Self::default();
        if alert > 0.5 {
            t.alert_turns = alert.round().clamp(2.0, 12.0) as u32;
        }
        if assist > 0.05 {
            t.assist_eta_scale = assist.clamp(0.5, 2.0);
        }
        t
    }
}

/// Minimum turns in ENGAGE before backing out (unless health is critical).
const MIN_ENGAGE_TURNS: u32 = 2;
/// An engaged enemy out of sight this many turns: back to ALERT.
const LOST_TARGET_TURNS: u32 = 2;
/// FINISH calls go out below this enemy health, and are answered from at most this far.
const FINISH_HP: f32 = 0.25;
const FINISH_MAX_ETA: i32 = 3;
/// Health fraction below which an engaged champion leaves even a fight it's winning.
const CRITICAL_HP: f32 = 0.15;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fsm {
    pub state: State,
    /// Round the current state was entered.
    pub since: u32,
    /// Fight-or-not, two turns of evidence to flip.
    pub fight: Ctr2,
    pub hist: History,
    /// Consecutive quiet turns in ALERT / turns the engaged enemy has been out of sight.
    quiet: u32,
    /// Consecutive turns hurt while disengaging (running isn't working).
    hit_running: u32,
    target: Option<Pos>,
}

impl Fsm {
    fn go(&mut self, s: State, round: u32) {
        if s != self.state {
            self.state = s;
            self.since = round;
            self.quiet = 0;
            self.hit_running = 0;
        }
    }

    fn turns_in_state(&self, round: u32) -> u32 {
        round.saturating_sub(self.since)
    }

    /// Will we answer this call?
    fn answers(&self, inp: &Inputs, c: &CallIn, k: &Tuning) -> bool {
        if !k.answer_calls
            || c.age > 1
            || inp.hp_frac < k.assist_min_hp
            || c.my_eta <= 0
            || c.my_eta > k.max_help_eta
        {
            return false;
        }
        let budget = (time_left_turns(c.time_left) as f32 * k.assist_eta_scale).round() as i32;
        match c.intent {
            CallIntent::Abort => false,
            CallIntent::Finish => !inp.big_bag && c.my_eta <= FINISH_MAX_ETA,
            CallIntent::Fight => !inp.big_bag && c.my_eta <= budget,
            // Covering a retreat is on the way home: allowed with a bag, if it's close.
            CallIntent::Escape => c.my_eta <= budget && (!inp.big_bag || c.my_eta <= 4),
        }
    }

    /// One turn: update the history-driven counters, move the state machine, and return the
    /// directive plus the help call to broadcast (if any).
    pub fn step(&mut self, inp: &Inputs, k: &Tuning) -> (Directive, Option<Call>) {
        let round = inp.round;
        if !inp.alive {
            // Tell anyone on the way that this fight is over.
            let abort = matches!(self.state, State::Engage | State::Disengage).then(|| Call {
                pos: inp.me,
                intent: CallIntent::Abort,
                time_left: 0,
            });
            self.go(State::Economy, round);
            self.fight = Ctr2::default();
            return (Directive::Keep, abort);
        }
        // Fight counter: the advantage, unless history says this kind or this streak is bad.
        if let Some(e) = inp.enemy {
            let history_ok = !self.hist.bad_against(e.kind) && self.hist.recent_losses(3) < 3;
            self.fight.update(e.advantage >= inp.fight_threshold && history_ok);
        }
        let threat_near = inp.threat_eta.is_some_and(|t| t <= k.alert_eta);
        let in_reach = inp.enemy.is_some_and(|e| e.dist <= e.range + 1);

        match self.state {
            State::Economy => {
                // Only an enemy that can reach us soon matters; one across the map doesn't.
                if threat_near {
                    self.go(State::Alert, round);
                    return self.alert(inp, k);
                }
                if let Some(c) = inp.call.filter(|c| self.answers(inp, c, k)) {
                    return self.start_help(&c, round);
                }
                (Directive::Keep, None)
            }
            State::Alert => self.alert(inp, k),
            State::Engage => {
                let Some(e) = inp.enemy else {
                    self.quiet += 1;
                    if self.quiet >= LOST_TARGET_TURNS {
                        self.go(State::Alert, round);
                        return (Directive::NoChannel, None);
                    }
                    return (self.target.map_or(Directive::NoChannel, Directive::Engage), None);
                };
                self.quiet = 0;
                self.target = Some(e.pos);
                let dwell_done = self.turns_in_state(round) >= MIN_ENGAGE_TURNS;
                let losing = self.fight.strong_no()
                    || (inp.hp_frac < k.retreat_hp && !self.fight.strong_yes());
                let bag_risk = inp.big_bag && !self.fight.strong_yes();
                if inp.hp_frac < CRITICAL_HP || (dwell_done && (losing || bag_risk)) {
                    self.go(State::Disengage, round);
                    return self.disengage(inp, k);
                }
                let call = if e.hp_frac < FINISH_HP {
                    Some(Call { pos: e.pos, intent: CallIntent::Finish, time_left: 3 })
                } else if !self.fight.strong_yes() {
                    let tl = time_left_bucket(inp.survive_turns);
                    (tl > 0).then_some(Call {
                        pos: e.pos,
                        intent: CallIntent::Fight,
                        time_left: tl,
                    })
                } else {
                    None
                };
                (Directive::Engage(e.pos), call)
            }
            State::Disengage => self.disengage(inp, k),
            State::Recover => {
                if in_reach {
                    self.go(State::Disengage, round);
                    return self.disengage(inp, k);
                }
                if inp.hp_frac > 0.9 && inp.threat_eta.is_none() {
                    self.go(State::Economy, round);
                    return (Directive::Keep, None);
                }
                (Directive::Recover, None)
            }
            State::Assist | State::Escort => {
                // Our own enemy in sight: fight it with the counter like anyone else.
                if let Some(e) = inp.enemy {
                    if self.fight.yes() && !(self.state == State::Escort && inp.big_bag) {
                        self.go(State::Engage, round);
                        self.target = Some(e.pos);
                        return (Directive::Engage(e.pos), None);
                    }
                    if in_reach {
                        self.go(State::Disengage, round);
                        return self.disengage(inp, k);
                    }
                }
                match inp.call.filter(|c| self.answers(inp, c, k)) {
                    Some(c) if c.intent == CallIntent::Escape => {
                        self.state = State::Escort;
                        (Directive::Escort(c.escort_point), None)
                    }
                    Some(c) => {
                        self.state = State::Assist;
                        (Directive::Assist(c.pos), None)
                    }
                    // Call gone, aborted or out of reach: stay wary where the fight was.
                    None => {
                        self.go(State::Alert, round);
                        (Directive::NoChannel, None)
                    }
                }
            }
        }
    }

    fn start_help(&mut self, c: &CallIn, round: u32) -> (Directive, Option<Call>) {
        if c.intent == CallIntent::Escape {
            self.go(State::Escort, round);
            (Directive::Escort(c.escort_point), None)
        } else {
            self.go(State::Assist, round);
            (Directive::Assist(c.pos), None)
        }
    }

    /// ALERT: an enemy is around. Fight if the counter says so, back off if it can reach us
    /// and the counter says no, otherwise keep farming without channels until it's quiet.
    fn alert(&mut self, inp: &Inputs, k: &Tuning) -> (Directive, Option<Call>) {
        let round = inp.round;
        if let Some(e) = inp.enemy {
            if self.fight.yes() && !inp.big_bag {
                self.go(State::Engage, round);
                self.target = Some(e.pos);
                return (Directive::Engage(e.pos), None);
            }
            if e.dist <= e.range + 1 {
                self.go(State::Disengage, round);
                return self.disengage(inp, k);
            }
        }
        // Quiet = nothing can reach us soon (a far enemy in sight doesn't keep us on alert).
        if inp.threat_eta.is_some_and(|t| t <= k.alert_eta) {
            self.quiet = 0;
        } else {
            self.quiet += 1;
        }
        if self.quiet >= k.alert_turns {
            self.go(State::Economy, round);
            return (Directive::Keep, None);
        }
        if let Some(c) = inp.call.filter(|c| self.answers(inp, c, k)) {
            return self.start_help(&c, round);
        }
        (Directive::NoChannel, None)
    }

    /// DISENGAGE: walk away; recall once nothing can reach us during the channel; turn and
    /// fight when running clearly isn't working; call for cover when chased.
    fn disengage(&mut self, inp: &Inputs, k: &Tuning) -> (Directive, Option<Call>) {
        let round = inp.round;
        let Some(e) = inp.enemy else {
            if inp.threat_eta.is_none_or(|t| t > inp.recall_ticks + 1) {
                return self.got_away(inp, k, None);
            }
            return (Directive::Disengage, None);
        };
        self.hit_running = if inp.hurt { self.hit_running + 1 } else { 0 };
        if self.hit_running >= 2
            && e.dist <= e.range
            && e.advantage >= k.last_stand_advantage
            && !inp.big_bag
        {
            self.go(State::Engage, round);
            self.target = Some(e.pos);
            return (Directive::Engage(e.pos), None);
        }
        let chased = e.dist <= e.range + 2;
        let call = chased.then(|| {
            let tl = time_left_bucket(inp.survive_turns);
            let intent = if tl == 0 { CallIntent::Abort } else { CallIntent::Escape };
            Call { pos: inp.me, intent, time_left: tl }
        });
        if e.dist > e.range + 1 + inp.recall_ticks {
            return self.got_away(inp, k, call);
        }
        (Directive::Disengage, call)
    }

    /// Out of reach: go home only when hurt (or carrying a big bag), else stay wary nearby.
    fn got_away(
        &mut self,
        inp: &Inputs,
        k: &Tuning,
        call: Option<Call>,
    ) -> (Directive, Option<Call>) {
        if inp.hp_frac < k.recover_below_hp || inp.big_bag {
            self.go(State::Recover, inp.round);
            (Directive::Recover, call)
        } else {
            self.go(State::Alert, inp.round);
            (Directive::NoChannel, call)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enemy(dist: i32, advantage: f32) -> EnemyIn {
        EnemyIn { pos: Pos::new(10 + dist, 10), dist, range: 1, kind: 3, hp_frac: 0.8, advantage }
    }

    fn inputs(round: u32) -> Inputs {
        Inputs {
            round,
            alive: true,
            me: Pos::new(10, 10),
            hp_frac: 1.0,
            big_bag: false,
            enemy: None,
            threat_eta: None,
            hurt: false,
            survive_turns: 20.0,
            fight_threshold: 1.25,
            call: None,
            recall_ticks: 5,
        }
    }

    #[test]
    fn counter_needs_two_turns_to_flip() {
        let mut c = Ctr2::default();
        assert!(!c.yes());
        c.update(true);
        assert!(c.yes());
        c.update(true);
        c.update(false);
        assert!(c.yes(), "one bad turn doesn't flip a strong yes");
        c.update(false);
        assert!(!c.yes());
    }

    #[test]
    fn history_counts_recent_losses_and_bad_kinds() {
        let mut h = History::default();
        assert_eq!(h.recent_losses(3), 0);
        h.record(Some(4), 10.0, 50.0);
        h.record(Some(4), 10.0, 50.0);
        h.record(None, 0.0, 0.0); // nothing happened: not recorded
        assert_eq!(h.recent_losses(3), 2);
        assert!(h.bad_against(4));
        h.record(Some(4), 60.0, 10.0);
        assert_eq!(h.recent_losses(3), 2);
        assert!(!h.bad_against(4));
    }

    #[test]
    fn spotting_an_enemy_stops_farming_until_quiet() {
        let k = Tuning::default();
        let mut f = Fsm::default();
        let mut i = inputs(1);
        // In sight but far (can't reach us for 10 turns): keep farming.
        i.enemy = Some(enemy(11, 0.5));
        i.threat_eta = Some(10);
        assert_eq!(f.step(&i, &k).0, Directive::Keep);
        assert_eq!(f.state, State::Economy);
        // Close enough to reach us soon: alert.
        i.enemy = Some(enemy(6, 0.5));
        i.threat_eta = Some(5);
        assert_eq!(f.step(&i, &k).0, Directive::NoChannel);
        assert_eq!(f.state, State::Alert);
        // Out of sight: stays wary for alert_turns quiet turns, then farms again.
        for r in 2..(2 + k.alert_turns) {
            let (d, _) = f.step(&inputs(r), &k);
            if r < 1 + k.alert_turns {
                assert_eq!(d, Directive::NoChannel, "round {r}");
            }
        }
        assert_eq!(f.state, State::Economy);
    }

    #[test]
    fn engage_does_not_flip_flop() {
        let k = Tuning::default();
        let mut f = Fsm::default();
        let mut i = inputs(1);
        i.enemy = Some(enemy(3, 2.0));
        i.threat_eta = Some(2);
        f.step(&i, &k);
        assert_eq!(f.state, State::Engage);
        // One turn where the estimate dips: still engaged.
        i.round = 2;
        i.enemy = Some(enemy(2, 1.0));
        assert!(matches!(f.step(&i, &k).0, Directive::Engage(_)));
        i.round = 3;
        i.enemy = Some(enemy(1, 2.0));
        assert!(matches!(f.step(&i, &k).0, Directive::Engage(_)));
        assert_eq!(f.state, State::Engage);
    }

    #[test]
    fn losing_fight_disengages_and_calls_escape_when_chased() {
        let k = Tuning::default();
        let mut f = Fsm::default();
        let mut i = inputs(1);
        i.enemy = Some(enemy(2, 3.0));
        i.threat_eta = Some(1);
        f.step(&i, &k);
        f.step(
            &{
                i.round = 2;
                i
            },
            &k,
        );
        for r in 3..7 {
            i.round = r;
            i.enemy = Some(enemy(1, 0.3));
            i.hp_frac = 0.3;
            i.survive_turns = 6.0;
            f.step(&i, &k);
        }
        assert_eq!(f.state, State::Disengage);
        i.round = 8;
        i.big_bag = true;
        let (d, call) = f.step(&i, &k);
        assert_eq!(d, Directive::Disengage);
        assert_eq!(call.map(|c| c.intent), Some(CallIntent::Escape));
    }

    #[test]
    fn running_that_fails_turns_into_a_last_stand() {
        let k = Tuning::default();
        let mut f = Fsm { state: State::Disengage, ..Fsm::default() };
        let mut i = inputs(1);
        i.enemy = Some(enemy(1, 0.9));
        i.hurt = true;
        f.step(&i, &k);
        i.round = 2;
        let (d, _) = f.step(&i, &k);
        assert!(matches!(d, Directive::Engage(_)));
        // ...but never with a big bag (the killer takes it).
        let mut g = Fsm { state: State::Disengage, ..Fsm::default() };
        i.big_bag = true;
        g.step(&i, &k);
        assert_eq!(g.step(&i, &k).0, Directive::Disengage);
    }

    fn call(intent: CallIntent, time_left: u8, my_eta: i32) -> CallIn {
        CallIn {
            pos: Pos::new(30, 30),
            intent,
            time_left,
            age: 0,
            my_eta,
            escort_point: Pos::new(25, 25),
        }
    }

    #[test]
    fn answers_calls_by_intent_and_time() {
        let k = Tuning::default();
        let mut i = inputs(1);
        i.call = Some(call(CallIntent::Fight, 2, 4));
        let mut f = Fsm::default();
        assert_eq!(f.step(&i, &k).0, Directive::Assist(Pos::new(30, 30)));
        // Too far for the time the caller has.
        let mut g = Fsm::default();
        i.call = Some(call(CallIntent::Fight, 1, 9));
        assert_eq!(g.step(&i, &k).0, Directive::Keep);
        // Escape: go to the escort point, even with a bag if it's close.
        let mut h = Fsm::default();
        i.big_bag = true;
        i.call = Some(call(CallIntent::Escape, 2, 3));
        assert_eq!(h.step(&i, &k).0, Directive::Escort(Pos::new(25, 25)));
        // A big bag never joins a fight.
        let mut j = Fsm::default();
        i.call = Some(call(CallIntent::Fight, 3, 3));
        assert_eq!(j.step(&i, &k).0, Directive::Keep);
        // Finish only from close by.
        let mut m = Fsm::default();
        i.big_bag = false;
        i.call = Some(call(CallIntent::Finish, 3, 5));
        assert_eq!(m.step(&i, &k).0, Directive::Keep);
    }

    #[test]
    fn abort_turns_helpers_back() {
        let k = Tuning::default();
        let mut f = Fsm::default();
        let mut i = inputs(1);
        i.call = Some(call(CallIntent::Fight, 3, 4));
        f.step(&i, &k);
        assert_eq!(f.state, State::Assist);
        i.round = 2;
        i.call = Some(call(CallIntent::Abort, 0, 3));
        assert_eq!(f.step(&i, &k).0, Directive::NoChannel);
        assert_eq!(f.state, State::Alert);
    }

    #[test]
    fn dying_while_engaged_broadcasts_abort() {
        let k = Tuning::default();
        let mut f = Fsm { state: State::Engage, ..Fsm::default() };
        let mut i = inputs(5);
        i.alive = false;
        let (_, call) = f.step(&i, &k);
        assert_eq!(call.map(|c| c.intent), Some(CallIntent::Abort));
        assert_eq!(f.state, State::Economy);
    }

    #[test]
    fn tuning_zero_keeps_defaults() {
        assert_eq!(Tuning::from_learned(0.0, 0.0), Tuning::default());
        let t = Tuning::from_learned(9.4, 1.5);
        assert_eq!(t.alert_turns, 9);
        assert!((t.assist_eta_scale - 1.5).abs() < 1e-6);
    }
}
