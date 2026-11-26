//! The four subspaces of a matrix: among its inputs, the ones it answers to and the
//! ones it sends to nothing; among its outputs, the ones it can reach and the
//! directions it never reaches.

use super::*;
use crate::math::linalg::svd;
use glam::DVec3;

pub const DEF: SceneDef = SceneDef {
    id: "subspaces",
    title: "The four subspaces",
    group: "Linear algebra: equations",
    blurb: "What a matrix sees, what it ignores, what it can reach, and what it never can.",
    make: |_| Box::new(Subspaces::create()),
};

// ----- colours, each with one meaning ------------------------------------------------

/// The row space, and x's part in it.
const ROW: Color = pal::BLUE;
/// The null space, and x's part in it.
const NULL: Color = pal::PINK;
/// The column space.
const COLUMN: Color = pal::GREEN;
/// The left null space.
const LEFT: Color = pal::ORANGE;
/// x and Ax, and the inputs that give the same output as x.
const POINT: Color = pal::WHITE;

// ----- sizes -------------------------------------------------------------------------

/// How much of each space is drawn: the ball of this radius round its origin.
const R: f64 = 1.5;
/// A singular value counts as zero when it is below this share of the largest. The
/// entries are kept to about seven significant digits, and rounding them alone leaves
/// the singular value of an exact zero at about a ten-millionth of the largest: below
/// a millionth is what is left of a zero, not a small number.
const TOL: f64 = 1e-6;
/// Shorter than this (in the units of the spaces), an arrow is not drawn and a corner
/// not marked.
const SHORT: f64 = 0.06;

// ----- matrices ----------------------------------------------------------------------

type Matrix = [[f32; 3]; 3];

// (Each chosen so that the arrows of its bases are seen well from the side the scene
// is first seen from.)

/// Every input reaches somewhere different, and every output is reached.
const RANK3: Matrix = [[0.7, 0.0, -0.2], [0.0, 0.4, -0.7], [0.0, 0.8, 0.1]];
/// Column 2 is twice column 3, so x = (0, 1, −2) gives 0; row 3 is row 1 − row 2, so
/// no output has any part along (1, −1, −1).
const RANK2: Matrix = [[0.3, 0.6, 0.3], [0.6, 0.0, 0.0], [-0.3, 0.6, 0.3]];
/// Every row a multiple of (1, −1, −1), every column one of (1, 2, −1).
const RANK1: Matrix = [[0.2, -0.2, -0.2], [0.4, -0.4, -0.4], [-0.2, 0.2, 0.2]];
const RANK0: Matrix = [[0.0; 3]; 3];
/// The rank-2 matrix with its middle entry moved from 0 to 0.01.
const NEARLY: Matrix = [[0.3, 0.6, 0.3], [0.6, 0.01, 0.0], [-0.3, 0.6, 0.3]];

const PRESETS: [(&str, Matrix); 4] = [("rank 3", RANK3), ("rank 2", RANK2), ("rank 1", RANK1), ("rank 0", RANK0)];

/// The input the scene opens with.
const START_X: [f64; 3] = [0.75, 0.85, -0.36];

/// How x moves by itself in an experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Moves {
    Still,
    /// Along the null space and back, both ways.
    AlongNull,
    /// Its row part once round a circle in the row space.
    RoundRow,
}

/// An experiment: its button, what to look for, the matrix, x, how x moves, and
/// whether the inputs with x's output and the image of the ball of inputs are shown.
struct Trial {
    name: &'static str,
    look: &'static str,
    a: Matrix,
    x: [f64; 3],
    moves: Moves,
    same: bool,
    image: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        name: "slide $x$ along the null space",
        look: "$x$ slides along the white dashed line, which runs beside the null space (pink): its row part $x_r$ stays where it is and only its null part $x_n$ changes. $Ax$ does not move at all, since $A$ sends every null vector to $0$: $Ax = Ax_r$. The dashed line is every input with this output: $x_r$, plus anything in the null space.",
        a: RANK2,
        x: START_X,
        moves: Moves::AlongNull,
        same: true,
        image: false,
    },
    Trial {
        name: "go round the row space",
        look: "Now $x$'s row part $x_r$ goes once round a circle in the row space (blue), and $Ax$ goes round an ellipse in the column space (green). No two points of the circle land on the same point of the ellipse: from the row space to the column space nothing is lost, and every point of the column space comes from exactly one point of the row space.",
        a: RANK2,
        x: [0.8, 0.75, -0.25],
        moves: Moves::RoundRow,
        same: false,
        image: false,
    },
    Trial {
        name: "rank 1",
        look: "Every row of $A$ is a multiple of $(1, -1, -1)$, and every column a multiple of $(1, 2, -1)$. The row space and the column space are lines; the null space and the left null space are planes: $1 + 2 = 3$ on each side. Drag $x$ anywhere: $Ax$ stays on the green line, and the inputs with the same output fill a whole plane.",
        a: RANK1,
        x: [0.9, 0.3, -0.6],
        moves: Moves::Still,
        same: true,
        image: false,
    },
    Trial {
        name: "rank 3",
        look: "Nothing is ignored and everything is reached: the row space and the column space are all of space, and the null space and the left null space are only the origin. $3 + 0 = 3$ on each side. Each output comes from one input and no other.",
        a: RANK3,
        x: [-0.5, 0.3, 0.9],
        moves: Moves::Still,
        same: false,
        image: false,
    },
    Trial {
        name: "nearly rank 2",
        look: r"The rank-2 matrix with its middle entry moved from 0 to 0.01. Its smallest singular value $\sigma_3$ is now 0.0026: not zero, so the rank is 3, the null space is the origin alone, and every output can be reached. But the ball of inputs lands on a pancake (the white ellipses): to reach out along its thin side takes an input about 370 times longer than to reach as far along its widest.",
        a: NEARLY,
        x: START_X,
        moves: Moves::Still,
        same: false,
        image: true,
    },
];

/// Seconds an experiment's movement takes, and a change of matrix's fade.
const TOUR_TIME: f32 = 6.0;
const FADE_TIME: f32 = 0.45;
/// How quickly the points are drawn gliding to a new place (the time in which they
/// cover about two thirds of the way, in seconds).
const GLIDE: f32 = 0.07;
/// The same for the line from a dot to its name: as fast as the name itself glides
/// (`frame.rs`, `LABEL_GLIDE`).
const LEADER_GLIDE: f32 = 0.07;

// ----- the four subspaces ------------------------------------------------------------

/// The four subspaces of a matrix, from its singular value decomposition, in f64.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spaces {
    a: [[f64; 3]; 3],
    rank: usize,
    /// The singular values, largest first.
    s: [f64; 3],
    /// v[..rank]: an orthonormal basis of the row space; v[rank..]: of the null space.
    v: [DVec3; 3],
    /// u[..rank]: of the column space; u[rank..]: of the left null space. A vₖ = σₖ uₖ.
    u: [DVec3; 3],
}

/// Does the largest coordinate of `v` (the first, of two as large) point backwards?
fn leads_backwards(v: DVec3) -> bool {
    let big = v.abs().max_element();
    v.to_array().into_iter().find(|c| c.abs() >= big - 1e-9).is_some_and(|c| c < 0.0)
}

impl Spaces {
    fn of(m: &Matrix) -> Spaces {
        let mut a = [[0.0f64; 3]; 3];
        for (i, row) in m.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                // (a number typed in is finite, but nothing is lost by making sure)
                a[i][j] = if x.is_finite() { x as f64 } else { 0.0 };
            }
        }
        let flat: Vec<f64> = a.iter().flatten().copied().collect();
        let (uu, s, vv) = svd(&flat, 3, 3);
        let col = |q: &[f64], k: usize| DVec3::new(q[k], q[3 + k], q[6 + k]);
        let rank = (0..3).filter(|&k| s[k] > TOL * s[0]).count();
        let (mut v, mut u) = ([col(&vv, 0), col(&vv, 1), col(&vv, 2)], [col(&uu, 0), col(&uu, 1), col(&uu, 2)]);
        // Each direction is drawn with its largest coordinate forwards, so that the
        // same matrix always shows the same arrows. A pair of the row and the column
        // space turns round together, so that A vₖ = σₖ uₖ still holds.
        for k in 0..3 {
            if leads_backwards(v[k]) {
                v[k] = -v[k];
                if k < rank {
                    u[k] = -u[k];
                }
            }
            if k >= rank && leads_backwards(u[k]) {
                u[k] = -u[k];
            }
        }
        Spaces { a, rank, s: [s[0], s[1], s[2]], v, u }
    }

    fn apply(&self, x: DVec3) -> DVec3 {
        let a = &self.a;
        DVec3::new(
            a[0][0] * x.x + a[0][1] * x.y + a[0][2] * x.z,
            a[1][0] * x.x + a[1][1] * x.y + a[1][2] * x.z,
            a[2][0] * x.x + a[2][1] * x.y + a[2][2] * x.z,
        )
    }

    /// x's parts in the row space and in the null space: x = xᵣ + xₙ, at right angles.
    /// (Each part is found from the smaller of the two spaces, and the other is what
    /// is left, so that the two add up to x exactly.)
    fn split(&self, x: DVec3) -> (DVec3, DVec3) {
        match self.rank {
            0 => (DVec3::ZERO, x),
            1 => {
                let xr = self.v[0] * self.v[0].dot(x);
                (xr, x - xr)
            }
            2 => {
                let xn = self.v[2] * self.v[2].dot(x);
                (x - xn, xn)
            }
            _ => (x, DVec3::ZERO),
        }
    }

    /// The one input in the row space that A sends to the column space's part of b.
    fn back(&self, b: DVec3) -> DVec3 {
        (0..self.rank).map(|k| self.v[k] * (self.u[k].dot(b) / self.s[k])).sum()
    }

    /// The scale the outputs are drawn at: whole, unless A stretches some input of the
    /// ball out of the ball of outputs; then as much smaller as keeps every Ax in it.
    fn out_scale(&self) -> f64 {
        if self.s[0] > 1.0 {
            1.0 / self.s[0]
        } else {
            1.0
        }
    }
}

/// The input nearest `x` inside the ball.
fn held(x: DVec3) -> DVec3 {
    let l = x.length();
    if l > R {
        x * (R / l)
    } else if l.is_finite() {
        x
    } else {
        DVec3::ZERO
    }
}

fn ease(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// x on its way in an experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tour {
    /// x = xᵣ + `reach` sin(θ) `along`, θ going once round from `from`: out to one end
    /// of the dashed line, back past the start to the other end, and back.
    AlongNull { xr: DVec3, along: DVec3, reach: f64, from: f64 },
    /// x = xₙ + `radius` (cos θ e₁ + sin θ e₂), θ going once round from 0.
    RoundRow { xn: DVec3, e1: DVec3, e2: DVec3, radius: f64 },
}

impl Tour {
    fn at(&self, t: f32) -> DVec3 {
        let turn = std::f64::consts::TAU * ease(t as f64);
        match *self {
            Tour::AlongNull { xr, along, reach, from } => xr + along * (reach * (from + turn).sin()),
            Tour::RoundRow { xn, e1, e2, radius } => xn + (e1 * turn.cos() + e2 * turn.sin()) * radius,
        }
    }
}

/// Where x, its row part and Ax are drawn (they glide to a new place).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Drawn {
    x: DVec3,
    xr: DVec3,
    ax: DVec3,
}

// ----- where things go on the canvas -------------------------------------------------

/// The size of the text on the cards, and of a line of it.
const CARD_TEXT: f32 = 12.5;
const CARD_LINE: f32 = 18.0;
/// The size of the matrix's numbers.
const MATRIX_TEXT: f32 = 13.0;
const MATRIX_WIDEST: usize = 5;
/// The direction the scene is seen from at first.
const YAW: f32 = -0.95;
const PITCH: f32 = 0.3;
/// The tangent of half the camera's field of view.
const TAN_HALF_FOV: f32 = 0.3203;

/// Where the two balls, their cards, the matrix and the arrow between go on a canvas
/// of `view` logical pixels, `inset` of whose top lies under the application's
/// buttons: side by side, or (on a canvas taller than it is wide, a phone held
/// upright) one above the other.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    /// The middle of each ball on the screen, the inputs' first, and their radius.
    centre: [Vec2; 2],
    radius: f32,
    /// The card under each ball: its corners.
    card: [(Vec2, Vec2); 2],
    /// The matrix's top-left corner, and the arrow from the inputs to the outputs.
    matrix: Vec2,
    arrow: (Vec2, Vec2),
    /// Whether the canvas has room for words at all (the strip of it left beside the
    /// controls on a phone does not: it shows the picture alone).
    words: bool,
}

/// How tall a card is: on a narrow canvas a line for each subspace and one for the
/// count; on a wide one also a line for each subspace's basis.
fn card_height(narrow: bool) -> f32 {
    (if narrow { 3.0 } else { 5.0 }) * CARD_LINE + 12.0
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let inset = inset.min(view.y * 0.5);
        let words = view.x >= 300.0 && view.y >= 300.0;
        // one above the other on a canvas taller than it is wide (a phone held
        // upright), and on one nearly square wherever that makes the balls larger
        let stacked = Layout::arranged(view, inset, true, words);
        let side_by_side = Layout::arranged(view, inset, false, words);
        if view.x < view.y * 0.9 || stacked.radius > 1.03 * side_by_side.radius {
            stacked
        } else {
            side_by_side
        }
    }

    /// The layout one above the other (`narrow`) or side by side.
    fn arranged(view: Vec2, inset: f32, narrow: bool, words: bool) -> Layout {
        let card = card_height(narrow);
        let mat = matrix_size(3, 3, MATRIX_WIDEST, MATRIX_TEXT);
        if !words {
            // the two balls alone, filling what there is (the cards and the matrix
            // are left out, and put out of the way)
            let free = Vec2::new(view.x, view.y - inset).max(Vec2::ONE);
            let (radius, centre) = if narrow {
                let radius = (free.x * 0.42).min(free.y * 0.2).max(0.5);
                (radius, [Vec2::new(view.x * 0.5, inset + free.y * 0.27), Vec2::new(view.x * 0.5, inset + free.y * 0.73)])
            } else {
                let radius = (free.x * 0.2).min(free.y * 0.42).max(0.5);
                (radius, [Vec2::new(view.x * 0.27, inset + free.y * 0.5), Vec2::new(view.x * 0.73, inset + free.y * 0.5)])
            };
            let nowhere = (Vec2::ZERO, Vec2::ZERO);
            return Layout { narrow, centre, radius, card: [nowhere; 2], matrix: Vec2::ZERO, arrow: (centre[0], centre[0]), words };
        }
        if narrow {
            // one above the other, the matrix beside the arrow between them
            let gap = mat.y + 22.0;
            let free = view.y - inset - 2.0 * card - gap - 40.0;
            let radius = (free * 0.25).min((view.x - 60.0) * 0.5).max(16.0);
            let x = view.x * 0.5;
            let c0 = Vec2::new(x, inset + 12.0 + radius);
            let card0 = (Vec2::new(8.0, c0.y + radius + 8.0), Vec2::new(view.x - 8.0, c0.y + radius + 8.0 + card));
            let below = card0.1.y + 6.0;
            let c1 = Vec2::new(x, below + gap + radius);
            let card1 = (Vec2::new(8.0, c1.y + radius + 8.0), Vec2::new(view.x - 8.0, c1.y + radius + 8.0 + card));
            let arrow_x = x - mat.x * 0.5 - 30.0;
            Layout {
                narrow: true,
                centre: [c0, c1],
                radius,
                card: [card0, card1],
                matrix: Vec2::new(arrow_x + 46.0, below + (gap - mat.y) * 0.5),
                arrow: (Vec2::new(arrow_x, below + 4.0), Vec2::new(arrow_x, below + gap - 4.0)),
                words,
            }
        } else {
            // side by side, the arrow between them and the matrix above it, where the
            // balls curve away: the gap is wide enough for that (on a small canvas,
            // wider than the arrow needs)
            let fit = |gap: f32| ((view.y - inset - card - 70.0) * 0.5).min((view.x - gap - 70.0) * 0.25).max(16.0);
            let gap = 120.0f32.max(212.0 - 0.43 * fit(120.0));
            let radius = fit(gap);
            let block = 2.0 * radius + 12.0 + card;
            let top = inset + 34.0f32.max((view.y - inset - block) * 0.45);
            let y = top + radius;
            let (x0, x1) = (view.x * 0.5 - gap * 0.5 - radius, view.x * 0.5 + gap * 0.5 + radius);
            let card_at = |x: f32| {
                let (lo, hi) = ((x - radius - 16.0).max(8.0), (x + radius + 16.0).min(view.x - 8.0));
                (Vec2::new(lo, y + radius + 12.0), Vec2::new(hi, y + radius + 12.0 + card))
            };
            Layout {
                narrow: false,
                centre: [Vec2::new(x0, y), Vec2::new(x1, y)],
                radius,
                card: [card_at(x0), card_at(x1)],
                matrix: Vec2::new(view.x * 0.5 - mat.x * 0.5 + 12.0, y - radius * 0.62 - mat.y),
                arrow: (Vec2::new(view.x * 0.5 - gap * 0.5 + 10.0, y), Vec2::new(view.x * 0.5 + gap * 0.5 - 10.0, y)),
                words,
            }
        }
    }

    /// The boxes on the screen that labels keep out of: the matrix with its name and
    /// rank, the arrow with its name, the headings, `inputs` and `heading` pixels wide
    /// (the cards are walls of their own).
    fn walls(&self, inputs: f32, heading: f32) -> Vec<(Vec2, Vec2)> {
        let mat = matrix_size(3, 3, MATRIX_WIDEST, MATRIX_TEXT);
        let (a, b) = self.arrow;
        let (r, [c0, c1]) = (self.radius, self.centre);
        let mut walls = vec![
            (self.matrix - Vec2::new(40.0, 4.0), self.matrix + mat + if self.narrow { Vec2::new(90.0, 4.0) } else { Vec2::new(4.0, 40.0) }),
            (a.min(b) - Vec2::splat(12.0), a.max(b) + Vec2::new(12.0, 34.0)),
            (c0 + Vec2::new(-r, -r) - Vec2::splat(4.0), c0 + Vec2::new(-r + inputs + 4.0, -r + 20.0)),
        ];
        let w = heading + 4.0;
        walls.push(if self.narrow { (c1 + Vec2::new(-r, -r) - Vec2::splat(4.0), c1 + Vec2::new(-r + w, -r + 20.0)) } else { (c1 + Vec2::new(r - w, -r - 4.0), c1 + Vec2::new(r + 4.0, -r + 20.0)) });
        walls
    }

    /// The camera's distance that makes a ball of radius `R` look `radius` pixels
    /// across on a canvas `height` pixels high (a little more in perspective).
    fn distance(&self, height: f32) -> f32 {
        let focal = height.max(1.0) * 0.5 / TAN_HALF_FOV;
        focal * R as f32 * 1.04 / self.radius.max(1.0)
    }
}

// ----- the scene ---------------------------------------------------------------------

pub struct Subspaces {
    a: Matrix,
    spaces: Spaces,
    x: DVec3,
    /// Where each handle is while it is dragged (the point itself may be held back:
    /// x to the ball, Ax to the column space): the inputs' and the outputs', in the
    /// ball's own units.
    grip: [Vec3; 2],
    /// An experiment's movement, and how far it has got (0 to 1).
    tour: Option<(Tour, f32)>,
    /// The picture being left after a change of matrix, and how far it has faded.
    fading: Option<(Spaces, f32)>,
    /// Counts the changes of matrix: the labels of a picture fading out are told from
    /// those of the one fading in by it.
    generation: u32,
    drawn: Drawn,
    /// Where the names of the subspaces that are the origin alone were put last frame,
    /// and the lines to them drawn (see `Leader`).
    leaders: std::cell::RefCell<Vec<Leader>>,
    /// What the names of the still picture hang on, as it was last frame (see
    /// `still_print`): while it stays the same, they stay where they are.
    still: std::cell::Cell<u64>,
    /// Where the names of the picture were last frame (see `Names`), and whether any
    /// was still gliding to a new place.
    spots: std::cell::RefCell<Vec<(Tag, Spot)>>,
    names_gliding: std::cell::Cell<bool>,
    /// How strongly each axis's name is drawn, and how strongly it is to be: a name
    /// that has no room fades out rather than vanishing, and back in when it has.
    axis_names: std::cell::Cell<[(f32, f32); 6]>,
    /// The names as they were written last, kept while nothing they are placed from
    /// changes (see `Still`): those of the still picture, which do not make way for x,
    /// and those of x and what moves with it.
    names: Still,
    marks: Still,
    /// What is written over the picture about the matrix (see `Words`).
    words: Words,
    /// The caption while x moves by itself: what it says, the bits of the width it has,
    /// and its lines.
    caption: std::cell::RefCell<(Vec<String>, u32, Vec<String>)>,
    show_bases: bool,
    show_same: bool,
    show_image: bool,
    /// The canvas last drawn on, and how much of its top the application's buttons cover.
    canvas: Vec2,
    inset: f32,
    new_view: bool,
}

impl Subspaces {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let spaces = Spaces::of(&RANK2);
        let x = DVec3::from_array(START_X);
        let mut s = Subspaces {
            a: RANK2,
            spaces,
            x,
            grip: [Vec3::ZERO; 2],
            tour: None,
            fading: None,
            generation: 0,
            drawn: Drawn { x, xr: x, ax: x },
            leaders: std::cell::RefCell::new(Vec::new()),
            still: std::cell::Cell::new(0),
            spots: std::cell::RefCell::new(Vec::new()),
            names_gliding: std::cell::Cell::new(false),
            axis_names: std::cell::Cell::new([(0.0, 0.0); 6]),
            names: Still::default(),
            marks: Still::default(),
            words: Words::default(),
            caption: std::cell::RefCell::new((Vec::new(), 0, Vec::new())),
            show_bases: true,
            show_same: true,
            show_image: false,
            canvas: Vec2::new(1070.0, 1000.0),
            inset: 0.0,
            new_view: false,
        };
        s.drawn = s.target();
        s
    }

    /// Where the points are, as opposed to where they are drawn.
    fn target(&self) -> Drawn {
        let (xr, _) = self.spaces.split(self.x);
        Drawn { x: self.x, xr, ax: self.spaces.apply(self.x) }
    }

    /// A new matrix. With `fade`, the picture of the old one fades out as the new one
    /// fades in (a change of rank always does: a plane does not turn into a line).
    fn set_matrix(&mut self, a: Matrix, fade: bool) {
        let old = self.spaces;
        self.a = a;
        self.spaces = Spaces::of(&a);
        if (fade || old.rank != self.spaces.rank) && old != self.spaces {
            // (a fade under way is left where it had got to: it starts from the picture as it is)
            self.fading = Some((old, 0.0));
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.set_matrix(t.a, true);
        self.x = DVec3::from_array(t.x);
        (self.show_bases, self.show_same, self.show_image) = (true, t.same, t.image);
        self.tour = self.tour_for(t.moves).map(|tour| (tour, 0.0));
        if let Some((tour, _)) = self.tour {
            self.x = tour.at(0.0);
        }
        self.new_view = true;
    }

    /// The movement that shows x moving in one way, from where it is.
    fn tour_for(&self, moves: Moves) -> Option<Tour> {
        let sp = &self.spaces;
        let (xr, xn) = sp.split(self.x);
        match moves {
            Moves::Still => None,
            Moves::AlongNull => {
                if sp.rank == 3 {
                    return None;
                }
                let along = xn.try_normalize().unwrap_or(sp.v[sp.rank]);
                // out to near the edge of the ball, on the line through xᵣ
                let reach = 0.92 * (R * R - xr.length_squared()).max(0.0).sqrt();
                let from = (xn.length() / reach.max(1e-12)).clamp(-1.0, 1.0).asin();
                Some(Tour::AlongNull { xr, along, reach, from })
            }
            Moves::RoundRow => {
                if sp.rank < 2 {
                    return None;
                }
                let e1 = xr.try_normalize().unwrap_or(sp.v[0]);
                let other = if e1.dot(sp.v[0]).abs() < 0.9 { sp.v[0] } else { sp.v[1] };
                let e2 = (other - e1 * e1.dot(other)).try_normalize()?;
                Some(Tour::RoundRow { xn, e1, e2, radius: xr.length() })
            }
        }
    }

    /// Move an experiment's tour and a fade on: the tour by `dt` seconds of the
    /// scene's time, the fade by `real_dt` of the clock's (it answers the hand).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if let Some((tour, t)) = self.tour.as_mut() {
            *t = (*t + dt.max(0.0) / TOUR_TIME).min(1.0);
            self.x = tour.at(*t);
            if *t >= 1.0 {
                self.tour = None;
            }
        }
        if let Some((_, t)) = self.fading.as_mut() {
            *t += real_dt.max(0.0) / FADE_TIME;
            if *t >= 1.0 || dt <= 0.0 {
                self.fading = None;
            }
        }
    }

    /// Bring the points as drawn `dt` seconds nearer to where they are; with
    /// `at_once`, all the way.
    fn glide(&mut self, dt: f32, at_once: bool) {
        let to = self.target();
        let k = 1.0 - (-(dt.max(0.0) / GLIDE) as f64).exp();
        let d = &mut self.drawn;
        for (drawn, to) in [(&mut d.x, to.x), (&mut d.xr, to.xr), (&mut d.ax, to.ax)] {
            *drawn += (to - *drawn) * k;
            // (exactly there in the end: the picture then stops changing)
            if at_once || !drawn.is_finite() || (to - *drawn).length() < 1e-5 {
                *drawn = to;
            }
        }
    }

    /// x moved so that its output is `b` as nearly as it can be: its row part is the
    /// one input of the row space that reaches b's part in the column space; its null
    /// part is kept, as far as the ball has room for it.
    fn reach_for(&self, b: DVec3) -> DVec3 {
        let (_, xn) = self.spaces.split(self.x);
        let mut xr = self.spaces.back(b);
        if !xr.is_finite() {
            return self.x;
        }
        if xr.length() > 0.98 * R {
            xr *= 0.98 * R / xr.length();
        }
        let room = (R * R - xr.length_squared()).max(0.0).sqrt();
        let xn = if xn.length() > room { xn * (room / xn.length()) } else { xn };
        xr + xn
    }

    fn layout(&self) -> Layout {
        Layout::of(self.canvas, self.inset)
    }

    /// Everything the still picture and its names are drawn from (see `Still`): not x
    /// and what moves with it, nor what is remembered of the names themselves.
    fn picture_inputs(&self, k: &mut Inputs) {
        let spaces = |k: &mut Inputs, sp: &Spaces| {
            k.f64s(sp.a.as_flattened()).f64s(&sp.s).int(sp.rank as u64);
            for v in sp.v.iter().chain(&sp.u) {
                k.f64s(&v.to_array());
            }
        };
        k.f32s(self.a.as_flattened());
        spaces(k, &self.spaces);
        match &self.fading {
            None => {
                k.int(0);
            }
            Some((old, t)) => {
                spaces(k.int(1).f32s(&[*t]), old);
            }
        }
        k.int(self.generation as u64).int(self.show_bases as u64).int(self.show_image as u64);
        k.f32s(&[self.canvas.x, self.canvas.y, self.inset]);
    }

    /// Everything the whole picture and its names are drawn from.
    fn inputs(&self, k: &mut Inputs) {
        self.picture_inputs(k);
        k.f64s(&self.x.to_array()).f32s(&self.grip[0].to_array()).f32s(&self.grip[1].to_array());
        match self.tour {
            None => k.int(0),
            Some((Tour::AlongNull { xr, along, reach, from }, t)) => k.int(1).f64s(&xr.to_array()).f64s(&along.to_array()).f64s(&[reach, from]).f32s(&[t]),
            Some((Tour::RoundRow { xn, e1, e2, radius }, t)) => k.int(2).f64s(&xn.to_array()).f64s(&e1.to_array()).f64s(&e2.to_array()).f64s(&[radius]).f32s(&[t]),
        };
        let d = self.drawn;
        k.f64s(&d.x.to_array()).f64s(&d.xr.to_array()).f64s(&d.ax.to_array()).int(self.show_same as u64);
    }

    fn the_view(&self) -> View {
        View::Orbit { target: Vec3::ZERO, yaw: YAW, pitch: PITCH, dist: self.layout().distance(self.canvas.y) }
    }

    /// Where the middle of each ball is in the scene. They are set off from the
    /// origin along the camera's own right and up directions, so that each stays in
    /// its place on the screen as the camera turns: the same turn shows both spaces
    /// from the same side, each turning about its own middle.
    fn centres(&self, f: &Frame) -> [Vec3; 2] {
        let lay = self.layout();
        let per_px = lay.distance(self.canvas.y) * TAN_HALF_FOV / (self.canvas.y.max(1.0) * 0.5);
        let mid = self.canvas * 0.5;
        lay.centre.map(|c| {
            let off = (c - mid) * per_px;
            f.cam.right * off.x - f.cam.up * off.y
        })
    }
}

impl Scene for Subspaces {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    /// Only an experiment's movement, a fade and a glide move by themselves.
    fn animated(&self) -> bool {
        self.tour.is_some() || self.fading.is_some() || self.drawn != self.target() || self.names_gliding.get() || self.axis_names.get().iter().any(|(now, to)| now != to)
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A matrix $A$ takes an input $x$ (in the first ball) to an output $Ax$ (in the second), and splits each side into two parts at right angles. Among the inputs: the row space, which $A$ answers to, and the null space, which it sends to $0$. Among the outputs: the column space, everything $A$ can reach, and the left null space, the directions no output has any part of. Drag $x$, or $Ax$; drag elsewhere to turn both.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.name, trial.look) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Matrix");
        let mut a = self.a;
        let (mut changed, mut typed) = (false, false);
        egui::Grid::new("subspaces matrix").spacing([6.0, 4.0]).show(ui, |ui| {
            for row in a.iter_mut() {
                for v in row.iter_mut() {
                    let r = entry(ui, v);
                    changed |= r.changed();
                    typed |= r.changed() && !r.dragged();
                }
                ui.end_row();
            }
        });
        if changed {
            self.set_matrix(a, typed);
            self.tour = None;
        }
        if let Some(k) = ui::buttons(ui, &PRESETS.map(|p| p.0)) {
            self.set_matrix(PRESETS[k].1, true);
            self.tour = None;
        }
        ui::note(ui, "Drag a number sideways, or click it and type. The buttons give a matrix of each rank.");

        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_bases, "a basis of each subspace");
        ui.checkbox(&mut self.show_same, ui::label_text(ui, "every input with the same output as $x$", 12.5, None, false));
        ui.checkbox(&mut self.show_image, "where the ball of inputs lands");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (ROW, "blue: the row space, and $x_r$, $x$'s part in it"),
                (NULL, "pink: the null space, and $x_n$, $x$'s part in it"),
                (COLUMN, "green: the column space"),
                (LEFT, "orange: the left null space"),
                (POINT, "white: $x$ and $Ax$ (drag either)"),
                (POINT, "white dashed: every input with the same output as $x$; while $x$ moves by itself, the paths of $x_r$ and $Ax$"),
                (POINT, "white ellipses (when shown): where the inputs in the ball land"),
                (POINT, "white corners: right angles"),
                (pal::DIM, "a sheet is a plane, a thick line a line, a tinted ball all of space, a ringed dot the origin alone; the paler arrows are a basis of each"),
                (pal::FAINT, "grey: the edge of the ball of each space that is drawn, and its axes"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Row space:", "every combination of the rows of $A$. $Ax$ is made of the dot products of $x$ with the rows, so only $x$'s part in the row space matters.");
        ui::term(ui, "Null space:", "every $x$ with $Ax = 0$: the inputs at right angles to every row.");
        ui::term(ui, "Column space:", "every combination of the columns of $A$, which is every output $Ax$.");
        ui::term(ui, "Left null space:", r"every $y$ at right angles to every column of $A$: no output $Ax$ has any part along it. (Left, because then $y$ times $A$ from the left, $y^\top A$, is $0$.)");
        ui::term(ui, "Rank:", "how many dimensions the row space has; the column space always has as many.");
        ui::term(ui, "Basis:", "as many directions as a space has dimensions, from which every vector in it is made in one way. Those drawn are at right angles and of length 1.");
        ui::term(ui, "Singular values:", r"$\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$, how much $A$ stretches its most and least stretched directions. The rank is how many are not zero.");

        ui::section(ui, "Numbers");
        let sp = &self.spaces;
        let (xr, xn) = sp.split(self.x);
        ui::readout(ui, "rank", sp.rank.to_string());
        for k in 0..3 {
            let v = sp.s[k];
            // (formulas all three, since a small one is written with a power of ten)
            let text = if k >= sp.rank && v > 0.0 { format!("${v:.1e}$, counts as 0") } else if v > 0.0 && v < 1e-3 { format!("${v:.2e}$") } else { format!("${}$", num(v, 4)) };
            ui::readout(ui, &format!(r"$\sigma_{}$", k + 1), text);
        }
        ui::note(ui, "A singular value below a millionth of the largest counts as 0: that is what rounding the entries can leave of an exact 0.");
        let point = |v: DVec3| format!("${}$", coords(v));
        ui::readout_colored(ui, "$x$", point(self.x), POINT);
        ui::readout_colored(ui, "$x_r$ (row part)", point(xr), ROW);
        ui::readout_colored(ui, "$x_n$ (null part)", point(xn), NULL);
        ui::readout_colored(ui, "$Ax$", point(sp.apply(self.x)), POINT);
        ui::readout(ui, "$Ax_r$", point(sp.apply(xr)));
        ui::readout(ui, "$Ax_n$", point(sp.apply(xn)));
        ui::readout(ui, "$|x_r|, |x|$", format!("{}, {}", num(xr.length(), 2), num(self.x.length(), 2)));
        ui::note(ui, "The bases drawn (at right angles, of length 1):");
        let r = sp.rank;
        for (color, letter, basis, from, to) in [(ROW, "v", sp.v, 0, r), (NULL, "n", sp.v, r, 3), (COLUMN, "u", sp.u, 0, r), (LEFT, "w", sp.u, r, 3)] {
            for k in from..to {
                ui::readout_colored(ui, &format!("${letter}_{}$", k - from + 1), point(basis[k]), color);
            }
        }
        if r == 3 {
            ui::note(ui, "The null space and the left null space are the origin alone: no basis vectors.");
        } else if r == 0 {
            ui::note(ui, "The row space and the column space are the origin alone: no basis vectors.");
        }
        // (two links in plain words: a link with a formula in it is set on one line, and
        // these would not fit across the panel)
        ui::see_also(ui, &[("elim", "finding the null space by elimination"), ("underdet", "the shortest of many solutions"), ("lsq", "the nearest output when $b$ cannot be reached"), ("svd", "the bases drawn here, and what $A$ does to the ball")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        if (view - self.canvas).abs().max_element() > 24.0 || f.top_inset != self.inset {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.new_view = true;
        }
        self.advance(f.dt, f.real_dt);
        // nothing glides while the scene's time stands still, nor while a handle is
        // dragged (a drag must be seen at once); by the clock, since it answers the hand
        let at_once = f.dt <= 0.0 || f.handles.dragging();
        self.glide(f.real_dt, at_once);
        let centres = self.centres(f);
        let k = self.spaces.out_scale();
        if !f.handles.dragging() {
            self.grip = [self.drawn.x.as_vec3(), (self.drawn.ax * k).as_vec3()];
        }
        let mut p = centres[0] + self.grip[0];
        if f.drag3(1, &mut p, None, POINT) {
            self.grip[0] = p - centres[0];
            self.x = held(self.grip[0].as_dvec3());
            self.tour = None;
            self.glide(0.0, true);
        }
        if self.spaces.rank > 0 {
            let mut p = centres[1] + self.grip[1];
            if f.drag3(2, &mut p, None, POINT) {
                self.grip[1] = p - centres[1];
                self.x = self.reach_for(self.grip[1].as_dvec3() / k);
                self.tour = None;
                self.glide(0.0, true);
            }
        }
        self.words.update(&self.a, &self.spaces, &self.layout());
        let (mut names, mut marks) = (std::mem::take(&mut self.names), std::mem::take(&mut self.marks));
        draw(f, self, centres, [&mut names, &mut marks]);
        (self.names, self.marks) = (names, marks);
    }
}

/// A vector's coordinates, to two places.
fn coords(v: DVec3) -> String {
    format!("({}, {}, {})", num(v.x, 2), num(v.y, 2), num(v.z, 2))
}

// ----- drawing -----------------------------------------------------------------------

/// One subspace as drawn: where its ball's middle is, how many dimensions it has, an
/// orthonormal basis (the first `dim` span it), its colour, how strongly it is drawn
/// (it fades in and out), its name, and the names of its basis vectors.
struct Piece {
    centre: Vec3,
    dim: usize,
    basis: [Vec3; 3],
    color: Color,
    alpha: f32,
    name: &'static str,
    letter: &'static str,
    /// How long a basis vector is drawn (the outputs may be drawn smaller).
    unit: f32,
    /// Which way on the screen its name prefers to go.
    side: Vec2,
}

/// The four pieces a matrix's picture is made of, faded to `alpha`.
fn pieces(sp: &Spaces, centres: [Vec3; 2], alpha: f32) -> [Piece; 4] {
    let r = sp.rank;
    let v = sp.v.map(|d| d.as_vec3());
    let u = sp.u.map(|d| d.as_vec3());
    let k = sp.out_scale() as f32;
    let rotate = |b: [Vec3; 3], by: usize| [b[by % 3], b[(by + 1) % 3], b[(by + 2) % 3]];
    // (the names of the row and column spaces go up and outwards, those of the null
    // spaces down)
    [
        Piece { centre: centres[0], dim: r, basis: v, color: ROW, alpha, name: "row space", letter: "v", unit: 1.0, side: Vec2::new(-1.0, if r == 3 { 1.0 } else { 0.4 }) },
        Piece { centre: centres[0], dim: 3 - r, basis: rotate(v, r), color: NULL, alpha, name: "null space", letter: "n", unit: 1.0, side: Vec2::new(0.2, 1.0) },
        Piece { centre: centres[1], dim: r, basis: u, color: COLUMN, alpha, name: "column space", letter: "u", unit: k, side: if r == 3 { Vec2::new(1.0, 1.0) } else { Vec2::new(0.4, -1.0) } },
        Piece { centre: centres[1], dim: 3 - r, basis: rotate(u, r), color: LEFT, alpha, name: "left null space", letter: "w", unit: k, side: Vec2::new(0.2, 1.0) },
    ]
}

/// The circle a ball's edge is seen as: its middle, the direction it faces and its radius.
fn silhouette(f: &Frame, c: Vec3, r: f32) -> (Vec3, Vec3, f32) {
    let to_eye = f.cam.eye - c;
    let d2 = to_eye.length_squared();
    // (an eye inside the ball, or a size that is not a number: the circle facing the eye)
    if d2.is_nan() || d2 <= r * r * 1.0001 {
        return (c, -f.cam.fwd, r);
    }
    (c + to_eye * (r * r / d2), to_eye.normalize_or_zero(), r * (1.0 - r * r / d2).sqrt())
}

/// Points round the circle a ball's edge is seen as, laid out from the camera's own
/// right direction, so that each keeps its place as the view turns. (A direction
/// found from the circle's facing alone, `any_orthonormal_vector`, swings round
/// whenever the view passes the level, and every point of the circle with it.)
fn edge(f: &Frame, c: Vec3, r: f32, n: usize) -> Vec<Vec3> {
    let (sc, normal, sr) = silhouette(f, c, r);
    let e1 = (f.cam.right - normal * normal.dot(f.cam.right)).try_normalize().unwrap_or(f.cam.right);
    ring(sc, e1, normal.cross(e1), sr, n)
}

/// Points round a circle in the plane of the unit directions `e1` and `e2`.
fn ring(c: Vec3, e1: Vec3, e2: Vec3, r: f32, n: usize) -> Vec<Vec3> {
    (0..n).map(|i| i as f32 / n as f32 * TAU).map(|t| c + (e1 * t.cos() + e2 * t.sin()) * r).collect()
}

/// The geometry of one subspace in its ball: a dot ringed for the origin alone, a
/// thick line for a line, a sheet for a plane, the ball tinted for all of space.
fn draw_piece(f: &mut Frame, room: &mut Room, p: &Piece) {
    let (c, r, a) = (p.centre, R as f32, p.alpha);
    match p.dim {
        0 => {
            f.dot(c, 11.0, p.color.a(a));
            // (a ring facing the eye, laid out from the camera's own directions: see `edge`)
            f.dl.lines.closed(&ring(c, f.cam.right, f.cam.up, 0.09, 32), 1.6, p.color.a(0.8 * a));
        }
        1 => {
            let d = p.basis[0];
            f.line(c - d * r, c + d * r, 3.2, p.color.a(0.95 * a));
            room.line(f, c - d * r, c + d * r);
        }
        2 => {
            let (e1, e2) = (p.basis[0], p.basis[1]);
            let rim = ring(c, e1, e2, r, 72);
            for i in 0..rim.len() {
                f.dl.tris_alpha.tri(c, rim[i], rim[(i + 1) % rim.len()], p.color.a(0.14 * a));
            }
            f.dl.lines.closed(&rim, 1.6, p.color.a(0.75 * a));
            for i in (0..rim.len()).step_by(4) {
                room.line(f, rim[i], rim[(i + 4) % rim.len()]);
            }
            // a square grid on it, cut off at the rim
            let lines = 3;
            for kk in -lines..=lines {
                let s = kk as f32 * r / (lines as f32 + 0.6);
                let half = (r * r - s * s).max(0.0).sqrt();
                for (along, across) in [(e1, e2), (e2, e1)] {
                    f.line(c + across * s - along * half, c + across * s + along * half, 1.0, p.color.a(0.3 * a));
                }
            }
        }
        _ => {
            let (sc, _, _) = silhouette(f, c, r);
            let rim = edge(f, c, r, 72);
            for i in 0..rim.len() {
                f.dl.tris_alpha.tri(sc, rim[i], rim[(i + 1) % rim.len()], p.color.a(0.11 * a));
            }
            f.dl.lines.closed(&rim, 1.8, p.color.a(0.7 * a));
            // three great circles, for the roundness
            for (e1, e2) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
                f.dl.lines.closed(&ring(c, e1, e2, r, 72), 1.0, p.color.a(0.25 * a));
            }
        }
    }
}

/// The basis arrows of a subspace (the first `dim` of its basis).
fn draw_basis(f: &mut Frame, room: &mut Room, p: &Piece) {
    for k in 0..p.dim {
        let tip = p.centre + p.basis[k] * p.unit;
        f.arrow(p.centre, tip, 2.2, p.color.mix(pal::WHITE, 0.25).a(p.alpha));
        room.line(f, p.centre, tip);
    }
}

/// The direction on the screen from `from` to `to` (zero if either is out of sight).
fn screen_dir(f: &Frame, from: Vec3, to: Vec3) -> Vec2 {
    match (on_screen(f, from), on_screen(f, to)) {
        (Some(p), Some(q)) => q - p,
        _ => Vec2::ZERO,
    }
}

/// How the names of a picture are placed (see `Names::beside`): looked for afresh;
/// kept where they were, as those of a picture that has not changed since the last
/// frame are (none of what their places hang on has: see `still_print`), which costs
/// next to nothing; or kept where they were and out of the way of the rest, as those
/// of the picture being left are while it fades (the new picture's names need not
/// make way for them, and then move again once they have gone).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Naming {
    Look,
    Keep,
    Fading,
}

/// One place a name can go: beside the point `point` of what it names, on its side
/// `away` (a unit direction on the screen).
#[derive(Clone, Copy, Debug)]
struct Place {
    point: Vec3,
    away: Vec2,
}

/// Where a name is: at which of its places, and where it was drawn, as an offset on
/// the screen from that place's point (it glides when it moves on).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spot {
    at: usize,
    drawn: Vec2,
}

/// How quickly a name glides to a new place (the time in which it covers about two
/// thirds of the way, in seconds): more slowly than labels usually do, since a plane
/// turning with the view can send its name some way round its edge.
const NAME_GLIDE: f32 = 0.16;
/// The fastest a name glides, in pixels a second.
const NAME_SPEED: f32 = 900.0;

/// The names' places from one frame to the next: last frame's spots, this frame's,
/// and whether any name is still on its way to a new place (then another frame is wanted).
struct Names<'a> {
    last: &'a [(Tag, Spot)],
    next: Vec<(Tag, Spot)>,
    gliding: bool,
    /// What writes the names down (see `Still`).
    still: &'a mut Still,
}

impl Names<'_> {
    /// Write `text` beside one of `places`. A name keeps to the place it had, which
    /// moves with what it names as the view turns or x moves, while that place has a
    /// little room. Otherwise it goes to the nearest clear place (a name new to the
    /// picture: the first clear one in the order `prefer` lists them), or failing that
    /// to the clearest (but only if that is clearly clearer than where it was), and
    /// glides there from where it was drawn.
    #[allow(clippy::too_many_arguments)]
    fn beside(&mut self, f: &mut Frame, room: &mut Room, naming: Naming, tag: Tag, places: &[Place], prefer: &[usize], text: &str, size: f32, color: Color, gap: f32) {
        let half = Vec2::new(label_width(f, text, size) * 0.5 + 4.0, size * 0.5 + 1.0);
        let screen: Vec<Option<Vec2>> = places.iter().map(|p| on_screen(f, p.point)).collect();
        let centre = |i: usize| {
            let a = places[i].away;
            screen[i].map(|p| p + a * (gap + (a.x * half.x).abs() + (a.y * half.y).abs()))
        };
        let was = self.last.iter().find(|(t, _)| *t == tag).map(|(_, s)| *s);
        let mut at = was.map(|s| s.at).filter(|&i| i < places.len() && screen[i].is_some());
        if naming == Naming::Look || (at.is_none() && naming == Naming::Keep) {
            let kept = at.and_then(|i| Some((room.room_for(screen[i]?, centre(i)?, half), i)));
            if !kept.is_some_and(|(r, _)| r >= 3.0) {
                // (a name that must move goes to the nearest clear place, not the
                // first it would choose afresh: a short way, that reads as the same name)
                let now = was.and_then(|w| Some(screen.get(w.at).copied().flatten()? + w.drawn));
                let order: Vec<usize> = match now {
                    Some(now) => {
                        let mut near: Vec<usize> = (0..places.len()).filter(|&i| screen[i].is_some()).collect();
                        near.sort_by(|&a, &b| centre(a).map_or(f32::MAX, |c| c.distance(now)).total_cmp(&centre(b).map_or(f32::MAX, |c| c.distance(now))));
                        near
                    }
                    None => prefer.to_vec(),
                };
                let mut best: Option<(f32, usize)> = None;
                for &i in order.iter().filter(|&&i| i < places.len()) {
                    let (Some(p), Some(c)) = (screen[i], centre(i)) else { continue };
                    let r = room.room_for(p, c, half);
                    if best.is_none_or(|b| r > b.0) {
                        best = Some((r, i));
                    }
                    if r >= 5.0 {
                        break;
                    }
                }
                at = match (kept, best) {
                    (Some((there, i)), Some((better, _))) if better < there + 3.0 => Some(i),
                    (_, Some((_, i))) => Some(i),
                    (kept, None) => kept.map(|(_, i)| i),
                };
            }
        }
        let Some(i) = at else { return };
        let (Some(p), Some(c)) = (screen[i], centre(i)) else { return };
        if naming != Naming::Fading {
            self.still.boxed(room, c - half, c + half);
        }
        // carried on from where it was drawn: a name that moves on glides there (by
        // the clock, as names do)
        let want = c - p;
        let from = match was {
            Some(w) if w.at == i => Some(w.drawn),
            Some(w) => screen.get(w.at).copied().flatten().map(|q| q + w.drawn - p),
            None => None,
        };
        let drawn = match from {
            Some(d) => {
                // (eased, and no faster than `NAME_SPEED`: a long way takes longer)
                let step = (want - d) * (1.0 - (-f.real_dt.max(0.0) / NAME_GLIDE).exp());
                let now = d + step.clamp_length_max(NAME_SPEED * f.real_dt.max(0.0));
                if (want - now).length() < 0.5 || !now.is_finite() {
                    want
                } else {
                    now
                }
            }
            None => want,
        };
        self.gliding |= drawn != want;
        self.still.text_px(f, p + drawn, text, size, color, Align2::CENTER_CENTER);
        self.next.push((tag, Spot { at: i, drawn }));
    }
}

/// How wide a name is drawn: one with mathematics in it as it is set (which costs a
/// look-up, the name being drawn in this frame as well), or from the fonts' metrics
/// before the fonts are in place; plain words estimated.
fn label_width(f: &Frame, text: &str, size: f32) -> f32 {
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    crate::typeset::measure(f.ctx, text, size).map_or_else(|| text_width(text, size), |set| set.x)
}

/// The places in the order of `score`, highest first.
fn ranked(places: &[Place], score: impl Fn(&Place) -> f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..places.len()).collect();
    order.sort_by(|&a, &b| score(&places[b]).total_cmp(&score(&places[a])));
    order
}

/// Places all round a ring of points, each on its side away from `middle` on the screen.
fn round(f: &Frame, middle: Vec3, points: &[Vec3]) -> Vec<Place> {
    points.iter().map(|&q| Place { point: q, away: screen_dir(f, middle, q).normalize_or_zero() }).collect()
}

/// Places along a line through `points`, on either side of it.
fn along(f: &Frame, a: Vec3, b: Vec3, points: &[Vec3]) -> Vec<Place> {
    let across = screen_dir(f, a, b).normalize_or_zero().perp();
    points.iter().flat_map(|&q| [Place { point: q, away: across }, Place { point: q, away: -across }]).collect()
}

/// The name of a subspace that is the origin alone, which stands a little way off its
/// dot with a line to it: where the name was put and where the line was drawn to, as
/// offsets on the screen from the dot in units of the ball's own size (so that both
/// keep their places as the view zooms). The line's end glides when the name moves on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Leader {
    tag: Tag,
    place: Vec2,
    end: Vec2,
}

/// The names of a subspace and of its basis vectors (see `Naming`).
#[allow(clippy::too_many_arguments)]
fn label_piece(f: &mut Frame, room: &mut Room, names: &mut Names, p: &Piece, generation: u32, bases: bool, naming: Naming, leaders: (&[Leader], &mut Vec<Leader>)) {
    if bases {
        label_basis(f, room, names, p, generation, naming);
        return;
    }
    let (c, r) = (p.centre, R as f32);
    let color = p.color.mix(pal::WHITE, 0.2).a(p.alpha);
    let side = p.side.normalize_or_zero();
    let tag = (p.name, generation);
    let frozen = naming != Naming::Look;
    match p.dim {
        0 => {
            // The origin alone is crowded with arrows: its name stands a little way
            // off, where there is room, with a short line to the dot.
            let Some(o) = on_screen(f, c) else { return };
            let half = Vec2::new(text_width(p.name, 14.0) * 0.5 + 4.0, 8.0);
            let unit = screen_dir(f, c, c + f.cam.right).length().max(1e-3);
            let (last, next) = leaders;
            let was = last.iter().find(|l| l.tag == Tag::from(tag)).copied();
            let place = if frozen {
                was.map(|l| l.place)
            } else {
                // where it was, while that still has a little room (it moves only
                // when it must); otherwise the first clear place, the side it
                // prefers first, nearer and then further off
                let kept = was.map(|l| (room.room_for(o, o + l.place * unit, half), l.place));
                let mut ways: Vec<Vec2> = (0..16).map(|q| Vec2::from_angle(q as f32 / 16.0 * TAU)).collect();
                ways.sort_by(|a, b| b.dot(side.normalize_or_zero()).total_cmp(&a.dot(side.normalize_or_zero())));
                let mut best = (f32::MIN, side.normalize_or_zero() * 0.4 * r);
                'find: for reach in [0.4 * r, 0.7 * r] {
                    if kept.is_some_and(|(room_there, _)| room_there >= 3.0) {
                        break;
                    }
                    for &d in &ways {
                        let at = (d * reach * unit + Vec2::new(d.x * half.x, d.y * half.y)) / unit;
                        let room_here = room.room_for(o, o + at * unit, half);
                        if room_here > best.0 {
                            best = (room_here, at);
                        }
                        if room_here >= 5.0 {
                            break 'find;
                        }
                    }
                }
                // (and where nothing is clear, it stays put unless somewhere is clearly better)
                Some(match kept {
                    Some((room_there, at)) if room_there >= 3.0 || best.0 < room_there + 3.0 => at,
                    _ => best.1,
                })
            };
            let Some(place) = place else { return };
            let centre = o + place * unit;
            if naming != Naming::Fading {
                names.still.boxed(room, centre - half, centre + half);
            }
            names.still.text_placed(f, tag.0, tag.1, o, centre - o, p.name, 14.0, color);
            // the line from the dot towards the name, stopping short of both; its end
            // glides as the name does when the name moves to a new place
            let end = match was {
                Some(l) if naming != Naming::Fading => {
                    let now = l.end + (place - l.end) * (1.0 - (-f.real_dt.max(0.0) / LEADER_GLIDE).exp());
                    if (place - now).length() * unit < 0.5 || !now.is_finite() {
                        place
                    } else {
                        now
                    }
                }
                Some(l) => l.end,
                None => place,
            };
            next.push(Leader { tag: Tag::from(tag), place, end });
            let end = end * unit;
            let (near, far) = (8.0 / end.length().max(1.0), 1.0 - (half.y + 3.0) / end.length().max(1.0));
            if far > near {
                names.still.hud_line(f, o + end * near, o + end * far, 1.0, color.a(0.6 * p.alpha));
            }
        }
        1 => {
            // beside the line near one of its ends, across it (along it, the name would
            // run into the line itself): the side and the end it prefers first
            let d = p.basis[0];
            let places = along(f, c - d, c + d, &[0.85, -0.85, 0.6, -0.6].map(|t| c + d * (t * r)));
            let prefer = ranked(&places, |q| q.away.dot(side) + 0.5 * screen_dir(f, c, q.point).normalize_or_zero().dot(side));
            names.beside(f, room, naming, Tag::from(tag), &places, &prefer, p.name, 14.0, color, 8.0);
        }
        2 => {
            // beside its rim, outwards
            // (a little further out than names usually stand; and where the rim runs
            // close under the edge of the ball, beyond that edge instead)
            let rim = ring(c, p.basis[0], p.basis[1], r, 24);
            let mut places = round(f, c, &rim);
            places.extend(round(f, c, &edge(f, c, r, 24)));
            let prefer = ranked(&places, |q| q.away.dot(side) + if (q.point - c).length() < 0.999 * r { 0.15 } else { 0.0 });
            names.beside(f, room, naming, Tag::from(tag), &places, &prefer, p.name, 14.0, color, 12.0);
        }
        _ => {
            // all of space: beside the tinted ball's edge, as seen (the edge is laid out
            // with the camera's own directions, so that its points stay put as it turns)
            let (sc, _, _) = silhouette(f, c, r);
            let places = round(f, sc, &edge(f, c, r, 16));
            let prefer = ranked(&places, |q| q.away.dot(side));
            names.beside(f, room, naming, Tag::from(tag), &places, &prefer, p.name, 14.0, color, 9.0);
        }
    }
}

/// The names of a subspace's basis vectors (see `label_piece`): beside the tip, the way
/// the arrow points first, or failing that beside its middle.
fn label_basis(f: &mut Frame, room: &mut Room, names: &mut Names, p: &Piece, generation: u32, naming: Naming) {
    let c = p.centre;
    let color = p.color.mix(pal::WHITE, 0.2).a(p.alpha);
    for k in 0..p.dim {
        let tip = c + p.basis[k] * p.unit;
        let out = screen_dir(f, c, tip).normalize_or_zero();
        let across = out.perp();
        let mid = c + p.basis[k] * (0.55 * p.unit);
        let ways = [out, (out + across).normalize_or_zero(), (out - across).normalize_or_zero(), across, -across];
        let mut places: Vec<Place> = ways.iter().map(|&away| Place { point: tip, away }).collect();
        places.extend([Place { point: mid, away: across }, Place { point: mid, away: -across }]);
        let prefer: Vec<usize> = (0..places.len()).collect();
        let name = format!("${}_{}$", p.letter, k + 1);
        names.beside(f, room, naming, Tag(p.letter, generation * 4 + k as u32), &places, &prefer, &name, 13.0, color, 6.0);
    }
}

/// x, its two parts at right angles and its output, alone (on a canvas with no room
/// for words).
fn draw_points(f: &mut Frame, s: &Subspaces, centres: [Vec3; 2]) {
    let d = s.drawn;
    let k = s.spaces.out_scale();
    let (cx, cb) = (centres[0], centres[1]);
    let (xr, x, ax) = (d.xr.as_vec3(), d.x.as_vec3(), (d.ax * k).as_vec3());
    if xr.length() > 1e-9 {
        f.arrow(cx, cx + xr, 2.8, ROW);
    }
    if (x - xr).length() > 1e-9 {
        f.arrow(cx + xr, cx + x, 2.8, NULL);
    }
    if ax.length() > 1e-9 {
        f.arrow(cb, cb + ax, 2.8, POINT);
    }
}

/// A summary of everything the places of the still picture's names hang on: the
/// camera, the canvas, the matrix and what is shown. (Not x: those names do not make
/// way for it.)
fn still_print(f: &Frame, s: &Subspaces, lay: &Layout) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let bits = f.cam.view_proj.to_cols_array().map(f32::to_bits);
    bits.hash(&mut h);
    [s.canvas.x, s.canvas.y, s.inset, f.px, lay.radius].map(f32::to_bits).hash(&mut h);
    s.a.iter().flatten().map(|v| v.to_bits()).collect::<Vec<u32>>().hash(&mut h);
    (s.generation, s.fading.is_some(), s.show_bases, s.show_image, lay.words).hash(&mut h);
    // (never 0, which is what nothing has been drawn yet reads as)
    h.finish() | 1
}

/// A dashed closed curve whose dashes are fixed to the curve itself (every other one
/// of its `points`' segments), so that they move with it as it moves, rather than
/// being measured afresh along the screen each frame.
fn dashed_ring(f: &mut Frame, room: &mut Room, points: &[Vec3], width: f32, color: Color) {
    let n = points.len();
    for i in (0..n).step_by(2) {
        f.line(points[i], points[(i + 1) % n], width, color);
    }
    // (for the names, the curve as a few longer chords: they cost less to keep clear of)
    for i in (0..n).step_by(4) {
        room.line(f, points[i], points[(i + 4) % n]);
    }
}

/// The whole picture. Its still parts and their names come first: the names of the
/// subspaces and their bases find their places among those alone, so that they stay
/// put while x moves; then x, Ax and their names, which keep clear of everything.
fn draw(f: &mut Frame, s: &Subspaces, centres: [Vec3; 2], [still, marks]: [&mut Still; 2]) {
    let lay = s.layout();
    let sp = &s.spaces;
    let k = sp.out_scale() as f32;
    let r = R as f32;
    let mut room = Room::new(f);
    for (lo, hi) in lay.card {
        room.wall(lo, hi);
    }
    for (lo, hi) in lay.walls(s.words.widths[0], s.words.widths[1]) {
        room.wall(lo, hi);
    }
    let (cx, cb) = (centres[0], centres[1]);
    // over the picture, and first in the list of what is written (so that a name that
    // comes or goes in the picture does not move its neighbours in the list): the
    // matrix, the arrow, the headings and the cards
    if lay.words {
        draw_hud(f, s, &lay);
    }

    // each space's ball, and its axes
    for &c in &centres {
        f.dl.lines.closed(&edge(f, c, r, 96), 1.3, pal::FAINT);
        let rim = edge(f, c, r, 18);
        for i in 0..rim.len() {
            room.line(f, rim[i], rim[(i + 1) % rim.len()]);
        }
        for d in Vec3::AXES {
            f.line(c - d * r, c + d * r, 1.0, pal::FAINT.a(0.7));
            room.line(f, c - d * r, c + d * r);
        }
    }

    // the four subspaces: the picture being left, fading out, and the new one
    let fade = s.fading.map(|(_, t)| smoothstep(t));
    let new = pieces(sp, centres, fade.unwrap_or(1.0));
    let old = s.fading.map(|(old, t)| pieces(&old, centres, 1.0 - smoothstep(t)));
    // (the picture being left is not in the way of the new one's names: they take
    // their places at once, and need not move again when it has gone)
    let mut elsewhere = Room::new(f);
    for p in old.iter().flatten() {
        draw_piece(f, &mut elsewhere, p);
    }
    for p in new.iter() {
        draw_piece(f, &mut room, p);
    }
    // the right angles between each pair, at the origin
    for pair in [Some(&new), old.as_ref()].into_iter().flatten() {
        for (a, b) in [(&pair[0], &pair[1]), (&pair[2], &pair[3])] {
            if a.dim > 0 && b.dim > 0 && a.dim < 3 && b.dim < 3 {
                let size = 0.17 * a.unit.max(0.6);
                right_angle(f, a.centre, a.basis[0], b.basis[0], size, pal::WHITE.a(0.8 * a.alpha));
            }
        }
    }
    if s.show_bases {
        for p in old.iter().flatten() {
            draw_basis(f, &mut elsewhere, p);
        }
        for p in new.iter() {
            draw_basis(f, &mut room, p);
        }
    }

    // where the ball of inputs lands: the ellipsoid with half-axes σₖ uₖ times the
    // ball's radius (flat where a σ is 0)
    let mut image_at: Vec<Vec3> = Vec::new();
    if s.show_image && sp.s[0] > 0.0 {
        let u = sp.u.map(|v| v.as_vec3());
        // (a σ that counts as zero is drawn as one: the ellipse is flat)
        let sg: [f32; 3] = std::array::from_fn(|k| if k < sp.rank { sp.s[k] as f32 } else { 0.0 });
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let pts: Vec<Vec3> = (0..96).map(|q| q as f32 / 96.0 * TAU).map(|t| cb + (u[i] * (sg[i] * t.cos()) + u[j] * (sg[j] * t.sin())) * (r * k)).collect();
            f.dl.lines.closed(&pts, 1.2, POINT.a(0.55));
            for q in (0..pts.len()).step_by(6) {
                room.line(f, pts[q], pts[(q + 6) % pts.len()]);
            }
        }
        // (its name goes beside the widest of the three, the lower left first)
        image_at = (0..12).map(|q| q as f32 / 12.0 * TAU).map(|t| cb + (u[0] * (sg[0] * t.cos()) + u[1] * (sg[1] * t.sin())) * (r * k)).collect();
        image_at.sort_by(|a, b| screen_dir(f, cb, *b).normalize_or_zero().dot(Vec2::new(-1.0, 1.0)).total_cmp(&screen_dir(f, cb, *a).normalize_or_zero().dot(Vec2::new(-1.0, 1.0))));
    }

    if !lay.words {
        // a canvas too small for words has the picture alone: x, its parts and Ax
        draw_points(f, s, centres);
        return;
    }

    // the names of all that: looked for afresh only when something they hang on has
    // changed since the last frame
    let print = still_print(f, s, &lay);
    let last_print = s.still.get();
    let naming = if last_print == print { Naming::Keep } else { Naming::Look };
    let last_spots = s.spots.take();
    let last = s.leaders.take();
    let mut next = Vec::new();
    // (written as they were in the last frame if nothing they are placed from has changed
    // since, and they had come to rest: see `Still`)
    s.picture_inputs(still.begin(f));
    let again = still.settle(f);
    let mut names = Names { last: &last_spots, next: Vec::new(), gliding: false, still };
    let still_spot = |(tag, _): &&(Tag, Spot)| *tag != Tag::from("same");
    if again {
        names.still.replay(f, &mut room);
        names.next.extend(last_spots.iter().filter(still_spot).copied());
        next.clone_from(&last);
    } else {
        s.still.set(print);
        // (the subspaces' names before their bases': the names matter more, and their
        // places then do not hang on where the small names went)
        for bases in [false, true].into_iter().filter(|b| !b || s.show_bases) {
            for p in new.iter() {
                label_piece(f, &mut room, &mut names, p, s.generation, bases, naming, (&last, &mut next));
            }
            if let Some(old) = &old {
                for p in old.iter() {
                    label_piece(f, &mut room, &mut names, p, s.generation.wrapping_sub(1), bases, Naming::Fading, (&last, &mut next));
                }
            }
        }
        if !image_at.is_empty() {
            let places = round(f, cb, &image_at);
            let prefer = ranked(&places, |q| q.away.dot(Vec2::new(-0.7, 0.7)));
            names.beside(f, &mut room, naming, Tag::from("image"), &places, &prefer, "where the ball of inputs lands", 12.0, POINT.a(0.9), 8.0);
        }
        // (they have come to rest when none of them is on its way anywhere, and none has
        // moved since the last frame)
        let on_way = names.gliding || next.iter().any(|l| l.end != l.place);
        let moved = on_way || !names.next.iter().eq(last_spots.iter().filter(still_spot)) || next != last || print != last_print;
        names.still.end(f, moved);
    }
    let gliding = names.gliding;

    // now what moves: x, its parts and its output, as drawn
    let d = s.drawn;
    let (xr, xn) = (d.xr.as_vec3(), (d.x - d.xr).as_vec3());
    let x = d.x.as_vec3();
    let ax = (d.ax * k as f64).as_vec3();

    // every input with the same output as x: x's row part, plus anything in the null
    // space (the places its name may go, kept for when the names are written)
    let mut same_at: Vec<Vec3> = Vec::new();
    if s.show_same && (1..3).contains(&sp.rank) {
        let reach = (r * r - xr.length_squared()).max(0.0).sqrt();
        let a = fade.unwrap_or(1.0);
        if sp.rank == 2 {
            let n = sp.v[2].as_vec3();
            f.dashed(cx + xr - n * reach, cx + xr + n * reach, 1.6, POINT.a(0.8 * a));
            room.line(f, cx + xr - n * reach, cx + xr + n * reach);
            // (always in this order, the same end first: the place a name is kept at is
            // measured from the first, which must not jump from one end to the other)
            same_at = [0.8, 0.55, -0.8, -0.55].iter().map(|t| cx + xr + n * (t * reach)).collect();
        } else {
            let (n1, n2) = (sp.v[1].as_vec3(), sp.v[2].as_vec3());
            let rim = ring(cx + xr, n1, n2, reach, 96);
            for i in 0..rim.len() {
                f.dl.tris_alpha.tri(cx + xr, rim[i], rim[(i + 1) % rim.len()], POINT.a(0.05 * a));
            }
            dashed_ring(f, &mut room, &rim, 1.5, POINT.a(0.75 * a));
            same_at = rim.iter().step_by(8).copied().collect();
        }
    }

    // the path of an experiment's movement
    if let Some((Tour::RoundRow { xn: _, e1, e2, radius }, _)) = s.tour {
        let (e1, e2, rr) = (e1.as_vec3(), e2.as_vec3(), radius as f32);
        dashed_ring(f, &mut room, &ring(cx, e1, e2, rr, 96), 1.4, POINT.a(0.7));
        let (b1, b2) = (sp.apply(e1.as_dvec3() * radius), sp.apply(e2.as_dvec3() * radius));
        let ellipse: Vec<Vec3> = (0..96).map(|q| q as f64 / 96.0 * std::f64::consts::TAU).map(|t| cb + ((b1 * t.cos() + b2 * t.sin()) * k as f64).as_vec3()).collect();
        dashed_ring(f, &mut room, &ellipse, 1.4, POINT.a(0.7));
    }

    // x, split into its two parts at right angles
    if xr.length() as f64 > 1e-9 {
        f.arrow(cx, cx + xr, 2.8, ROW);
        room.line(f, cx, cx + xr);
    }
    if xn.length() as f64 > 1e-9 {
        f.arrow(cx + xr, cx + x, 2.8, NULL);
        room.line(f, cx + xr, cx + x);
    }
    if (xr.length() as f64) > SHORT && (xn.length() as f64) > SHORT {
        right_angle(f, cx + xr, -xr, xn, 0.13, pal::WHITE.a(0.85));
    }
    // and its output
    if ax.length() > 1e-9 {
        f.arrow(cb, cb + ax, 2.8, POINT);
        room.line(f, cb, cb + ax);
    }
    if sp.rank == 0 {
        f.dot(cb + ax, 12.0, POINT);
    }

    // their names: likewise written as they were if nothing has changed, the boxes the
    // names above keep the others out of among what they are placed from
    s.inputs(marks.begin(f));
    names.still.boxes(marks.inputs());
    let again = marks.settle(f);
    names.still = marks;
    if again {
        // (and what is remembered of them stays as it was)
        names.still.replay(f, &mut room);
        names.next.extend(last_spots.iter().filter(|spot| !still_spot(spot)).copied());
        s.names_gliding.set(gliding);
        s.spots.replace(names.next);
        s.leaders.replace(next);
        return;
    }
    names.gliding = false;
    let out = |f: &Frame, c: Vec3, p: Vec3| {
        let d = screen_dir(f, c, p);
        if d.length() > 1.0 {
            d
        } else {
            Vec2::new(1.0, -1.0)
        }
    };
    // (the dashed line's name first: x's names make way for it, not it for them,
    // so that it stays put as x slides along the line)
    if sp.rank == 2 && !same_at.is_empty() {
        // beside the dashed line, on the side away from the null space it runs beside
        // first, the ends before the middle
        let n = sp.v[2].as_vec3();
        let reach = (r * r - xr.length_squared()).max(0.0).sqrt();
        let spots: Vec<Vec3> = [0.8f32, -0.8, 0.65, -0.65, 0.5, -0.5, 0.35, -0.35, 0.2, -0.2].iter().map(|t| cx + xr + n * (t * reach)).collect();
        let places = along(f, cx + xr - n, cx + xr + n, &spots);
        let apart = screen_dir(f, cx, cx + xr);
        let prefer = ranked(&places, |q| if q.away.dot(apart) >= 0.0 { 1.0 } else { 0.0 });
        names.beside(f, &mut room, Naming::Look, Tag::from("same"), &places, &prefer, "same output as $x$", 12.5, POINT.a(0.9), 8.0);
    } else if !same_at.is_empty() {
        // (for a plane of them, outwards from its rim)
        let places = round(f, cx + xr, &same_at);
        let prefer = ranked(&places, |q| q.away.dot(Vec2::new(-0.7, -0.7)));
        names.beside(f, &mut room, Naming::Look, Tag::from("same"), &places, &prefer, "same output as $x$", 12.5, POINT.a(0.9), 8.0);
    }

    let dir = out(f, cx, cx + x);
    names.still.label(&mut room, f, "x", cx + x, "$x$", 17.0, POINT, dir, 10.0, false);
    names.still.label(&mut room, f, "Ax", cb + ax, "$Ax$", 17.0, POINT, out(f, cb, cb + ax), 10.0, false);
    if xr.length() as f64 > SHORT {
        let across = screen_dir(f, cx + xr, cx + x);
        names.still.label(&mut room, f, "xr", cx + xr * 0.55, "$x_r$", 15.0, ROW.mix(pal::WHITE, 0.25), -across, 8.0, false);
    }
    if xn.length() as f64 > SHORT {
        let across = screen_dir(f, cx + xr, cx);
        names.still.label(&mut room, f, "xn", cx + xr + xn * 0.5, "$x_n$", 15.0, NULL.mix(pal::WHITE, 0.25), -across, 8.0, false);
    }

    // the axes' names last, where there is room left: at the end of an axis that is
    // well out from the middle on the screen (one that points at the eye ends near the
    // middle, among everything else), and clear of the rest. One that has no room
    // fades out, and back in when it has (always written, faint or not, so that the
    // names of the picture do not change places in the list of what is written).
    let mut fades = s.axis_names.get();
    let radius = lay.radius;
    for (i, &c) in centres.iter().enumerate() {
        let names_of = if i == 0 { ["$x_1$", "$x_2$", "$x_3$"] } else { ["$b_1$", "$b_2$", "$b_3$"] };
        for (j, d) in Vec3::AXES.into_iter().enumerate() {
            let end = c + d * r;
            let out = screen_dir(f, c, end);
            let Some(at) = on_screen(f, end) else { continue };
            let half = Vec2::new(label_width(f, names_of[j], 12.0) * 0.5 + 4.0, 7.0);
            let centre = at + out.normalize_or_zero() * (6.0 + half.x);
            let (now, mut to) = fades[i * 3 + j];
            if naming == Naming::Look {
                to = if out.length() >= 0.75 * radius && room.room_for(at, centre, half) > 4.0 { 1.0 } else { 0.0 };
            }
            let now = if f.dt <= 0.0 { to } else { now + (to - now) * (1.0 - (-f.real_dt.max(0.0) / NAME_GLIDE).exp()) };
            let now = if (to - now).abs() < 0.02 { to } else { now };
            fades[i * 3 + j] = (now, to);
            if to > 0.5 {
                room.boxed(centre - half, centre + half);
            }
            names.still.text_px(f, centre, names_of[j], 12.0, pal::DIM.a(now), Align2::CENTER_CENTER);
        }
    }
    let on_way = names.gliding || fades.iter().any(|(now, to)| now != to);
    let moved = on_way || !names.next.iter().filter(|spot| !still_spot(spot)).eq(last_spots.iter().filter(|spot| !still_spot(spot))) || fades != s.axis_names.get();
    names.still.end(f, moved);
    s.axis_names.set(fades);
    s.names_gliding.set(gliding || names.gliding);
    s.spots.replace(names.next);
    s.leaders.replace(next);
}

fn draw_hud(f: &mut Frame, s: &Subspaces, lay: &Layout) {
    let sp = &s.spaces;
    // the arrow from the inputs to the outputs, and the matrix that does it
    let (a, b) = lay.arrow;
    hud_arrow(f, a, b, 2.0, pal::DIM);
    let mid = (a + b) * 0.5;
    let words = &s.words;
    let at = lay.matrix;
    let size = hud_matrix(f, at, &words.rows, MATRIX_WIDEST, MATRIX_TEXT, pal::DIM);
    f.text_px(at + Vec2::new(-8.0, size.y * 0.5), "$A =$", 14.0, pal::TEXT, Align2::RIGHT_CENTER);
    // the rank, and the singular values it was decided from
    if lay.narrow {
        // (the arrow is the matrix's own, on its left; only the smallest σ has room)
        f.text_px(at + Vec2::new(size.x + 10.0, size.y * 0.5 - 9.0), words.rank.as_str(), 13.0, pal::TEXT, Align2::LEFT_CENTER);
        f.text_px(at + Vec2::new(size.x + 10.0, size.y * 0.5 + 9.0), words.sigmas.as_str(), 11.5, pal::DIM, Align2::LEFT_CENTER);
    } else {
        f.text_px(mid + Vec2::new(0.0, 10.0), "$A$", 17.0, pal::TEXT, Align2::CENTER_TOP);
        f.text_px(at + Vec2::new(size.x * 0.5, size.y + 5.0), words.rank.as_str(), 13.0, pal::TEXT, Align2::CENTER_TOP);
        f.text_px(at + Vec2::new(size.x * 0.5, size.y + 22.0), words.sigmas.as_str(), 11.5, pal::DIM, Align2::CENTER_TOP);
    }

    // the headings, in the corners the balls leave free
    let r = lay.radius;
    let (c0, c1) = (lay.centre[0], lay.centre[1]);
    f.text_px(c0 + Vec2::new(-r, -r), INPUTS, 14.0, pal::TEXT, Align2::LEFT_TOP);
    if lay.narrow {
        f.text_px(c1 + Vec2::new(-r, -r), words.heading.as_str(), 14.0, pal::TEXT, Align2::LEFT_TOP);
    } else {
        f.text_px(c1 + Vec2::new(r, -r), words.heading.as_str(), 14.0, pal::TEXT, Align2::RIGHT_TOP);
    }

    // while x moves by itself, what to see (its lines worked out again only when what
    // it says or the room for it changes: they are measured, formulas and all)
    let mut kept = s.caption.borrow_mut();
    if let Some((tour, _)) = s.tour {
        let b = sp.apply(s.x);
        let parts: Vec<String> = match tour {
            Tour::AlongNull { .. } => vec!["$x$ slides along the null space:".into(), "$x_n$ changes, $x_r$ does not,".into(), format!("and $Ax$ stays at ${}$", coords(b))],
            Tour::RoundRow { .. } => vec!["$x_r$ goes round a circle in the row space,".into(), "and $Ax$ round an ellipse in the column space".into()],
        };
        // as few lines as the width allows
        let room = if lay.narrow { lay.card[1].1.x - lay.card[1].0.x - 20.0 } else { f.view_size().x - 40.0 };
        if !(kept.0 == parts && kept.1 == room.to_bits() && keeps()) {
            let mut lines: Vec<String> = Vec::new();
            for part in &parts {
                match lines.last_mut() {
                    Some(last) if text_width(last, 13.0) + text_width(part, 13.0) + 8.0 <= room => {
                        last.push(' ');
                        last.push_str(part);
                    }
                    _ => lines.push(part.clone()),
                }
            }
            *kept = (parts, room.to_bits(), lines);
        }
    }
    let caption: &[String] = if s.tour.is_some() { &kept.2 } else { &[] };
    if !lay.narrow {
        // centred under the cards
        let x = f.view_size().x * 0.5;
        for (i, line) in caption.iter().enumerate() {
            f.caption_px(Vec2::new(x, lay.card[0].1.y + 24.0 + 20.0 * i as f32), line.as_str(), 13.0, pal::TEXT, Align2::CENTER_TOP);
        }
    }

    // the cards: each subspace's dimension and basis, and the count (on a narrow
    // canvas, while x moves by itself, the outputs' card says what to see instead)
    for side in 0..2 {
        let (lo, hi) = lay.card[side];
        hud_panel(f, lo, hi);
        let at = |line: usize| lo + Vec2::new(10.0, 6.0 + CARD_LINE * (line as f32 + 0.5));
        if lay.narrow && side == 1 && !caption.is_empty() {
            for (line, text) in caption.iter().enumerate() {
                f.text_px(at(line), text.as_str(), 13.0, pal::TEXT, Align2::LEFT_CENTER);
            }
        } else {
            for (line, (text, size, color)) in words.cards[side].iter().enumerate() {
                f.text_px(at(line), text.as_str(), *size, *color, Align2::LEFT_CENTER);
            }
        }
    }
}

/// What the words over the picture were written for: the bits of the matrix's entries
/// (as typed, and as the subspaces were found from), whether the canvas is narrow, and
/// the bits of the cards' widths.
type WordsFor = ([[u32; 3]; 3], [[u64; 3]; 3], bool, [u32; 2]);

/// What is written over the picture about the matrix: its entries, its rank and
/// singular values, the outputs' heading and the cards. Written afresh only when the
/// matrix or the canvas's shape changes, not in every frame.
#[derive(Default)]
struct Words {
    made_for: Option<WordsFor>,
    rows: Vec<Vec<(String, Color)>>,
    rank: String,
    /// The singular values: on a narrow canvas only the smallest has room.
    sigmas: String,
    heading: String,
    /// How wide the two headings are, the inputs' and the outputs'.
    widths: [f32; 2],
    cards: [Vec<(String, f32, Color)>; 2],
}

impl Words {
    fn update(&mut self, a: &Matrix, sp: &Spaces, lay: &Layout) {
        let rooms = lay.card.map(|(lo, hi)| hi.x - lo.x - 20.0);
        let made_for = (a.map(|row| row.map(f32::to_bits)), sp.a.map(|row| row.map(f64::to_bits)), lay.narrow, rooms.map(f32::to_bits));
        if self.made_for == Some(made_for) && keeps() {
            return;
        }
        self.made_for = Some(made_for);
        self.rows = a.iter().map(|row| row.iter().map(|&v| (format!("${}$", num(v as f64, 2)), pal::TEXT)).collect()).collect();
        self.rank = format!("rank {}", sp.rank);
        let shown: Vec<String> = (0..3).map(|k| sigma(sp, k)).collect();
        self.sigmas = if lay.narrow { format!("$\\sigma_3 = {}$", shown[2]) } else { format!("$\\sigma = {}$", shown.join(", ")) };
        self.heading = outputs_heading(sp.out_scale());
        self.widths = [text_width(INPUTS, 14.0), text_width(&self.heading, 14.0)];
        self.cards = [0, 1].map(|side| card(sp, side, lay.narrow, rooms[side]));
    }
}

const INPUTS: &str = "inputs $x$";

/// Singular value `k` as written over the picture: "0" for one that counts as zero,
/// and two significant digits for a small one (0.0026, not 0.003).
fn sigma(sp: &Spaces, k: usize) -> String {
    let v = sp.s[k];
    if k >= sp.rank {
        "0".to_string()
    } else if v >= 0.1 {
        num(v, 3)
    } else {
        num(v, (1.0 - v.log10().floor()).clamp(0.0, 12.0) as usize)
    }
}

/// The heading of the outputs' ball, which says if they are drawn smaller.
fn outputs_heading(scale: f64) -> String {
    if scale < 1.0 {
        format!("outputs $Ax$, drawn {} times smaller", num(1.0 / scale, 1))
    } else {
        "outputs $Ax$".to_string()
    }
}

/// What a space's dimension makes a subspace of it.
fn what(dim: usize) -> &'static str {
    match dim {
        0 => "the origin alone",
        1 => "a line",
        2 => "a plane",
        _ => "all of space",
    }
}

/// The lines of the card under the inputs (`side` 0) or the outputs (1), no wider
/// than `room` pixels: each subspace with its dimension and (on a wide canvas) its
/// basis, and the count. Each line with its size and colour.
fn card(sp: &Spaces, side: usize, narrow: bool, room: f32) -> Vec<(String, f32, Color)> {
    let r = sp.rank;
    let (parts, basis) = if side == 0 { ([(ROW, "row space", "v", r, 0), (NULL, "null space", "n", 3 - r, r)], sp.v) } else { ([(COLUMN, "column space", "u", r, 0), (LEFT, "left null space", "w", 3 - r, r)], sp.u) };
    let mut lines = Vec::new();
    for (color, name, letter, dim, from) in parts {
        let color = color.mix(pal::WHITE, 0.2);
        lines.push((format!("{name}: {}, dimension {dim}", what(dim)), CARD_TEXT, color));
        if !narrow {
            let vectors: Vec<String> = (0..dim).map(|q| format!("${}_{} = {}$", letter, q + 1, coords(basis[from + q]))).collect();
            let text = if dim == 0 { "no basis vectors: $0$ is the only vector in it".to_string() } else { format!("basis {}", vectors.join(", ")) };
            // (in the smaller type if need be; failing that, where to find the numbers)
            let (text, size) = match [11.5, 10.5].into_iter().find(|&s| text_width(&text, s) + 12.0 <= room) {
                Some(size) => (text, size),
                None => (format!("basis ${}$: the numbers are in the panel", (0..dim).map(|q| format!("{letter}_{}", q + 1)).collect::<Vec<_>>().join(", ")), 11.5),
            };
            lines.push((format!("   {text}"), size, color.a(0.85)));
        }
    }
    let total = format!("${} + {} = 3$: every {} splits into a part in each, at right angles", r, 3 - r, if side == 0 { "input" } else { "output" });
    let total = if text_width(&total, CARD_TEXT) <= room { total } else { format!("${} + {} = 3$", r, 3 - r) };
    lines.push((total, CARD_TEXT, pal::TEXT));
    lines
}

// ----- labels kept from frame to frame ----------------------------------------------------
// (Rotations, elimination and span use these too.)

/// The bits of every number that a frame's labels are made from, in order. Two frames
/// with the same list were given the same inputs: compared bit for bit, not within a
/// tolerance, since a label placed for a number a millionth away may go elsewhere.
#[derive(Default, PartialEq)]
pub(super) struct Inputs(Vec<u32>);

impl Inputs {
    pub(super) fn f32s(&mut self, v: &[f32]) -> &mut Inputs {
        self.0.extend(v.iter().map(|x| x.to_bits()));
        self
    }

    pub(super) fn f64s(&mut self, v: &[f64]) -> &mut Inputs {
        self.0.extend(v.iter().flat_map(|x| [x.to_bits() as u32, (x.to_bits() >> 32) as u32]));
        self
    }

    pub(super) fn int(&mut self, v: u64) -> &mut Inputs {
        self.0.extend([v as u32, (v >> 32) as u32]);
        self
    }

    /// Everything a frame gives a scene that its labels can depend on: the camera, the
    /// canvas, the pixel ratio, the strip under the application's buttons.
    fn frame(&mut self, f: &Frame) {
        let c = f.cam;
        self.f32s(&c.view_proj.to_cols_array()).f32s(&c.inv_view_proj.to_cols_array()).f32s(&c.view.to_cols_array()).f32s(&c.proj.to_cols_array());
        for v in [c.eye, c.right, c.up, c.fwd] {
            self.f32s(&v.to_array());
        }
        self.f32s(&[c.viewport.x, c.viewport.y, c.near, c.fov_y, f.px, f.top_inset]).int(c.is_flat() as u64);
        for v in f.hud.iter().flatten() {
            self.f32s(&v.to_array());
        }
    }
}

/// One thing a frame's labels wrote.
#[derive(Clone, Debug, PartialEq)]
enum Took {
    /// Text written where it is (`Frame::text_px`, `Frame::caption_px`).
    Text { pos: Vec2, text: String, size: f32, color: Color, align: Align2, backed: bool },
    /// Text that keeps its place from frame to frame (`Frame::text_placed`): its name,
    /// the point on the screen it is placed from, and its place.
    Placed { name: &'static str, number: u32, from: Vec2, offset: Vec2, text: String, size: f32, color: Color },
    /// A line over the picture (`hud_line`), such as one from a name to what it names.
    Line(Vec2, Vec2, f32, Color),
    /// A box the labels written later keep out of (`Room::boxed`).
    Boxed(Vec2, Vec2),
}

/// A group of labels as they were last written, kept with what they were made from.
/// Placing labels is most of what these scenes cost in a frame (each place tried is
/// measured against every line drawn), and in a frame whose inputs are all what they were
/// in the last, it would only find the same places again: then the labels are written
/// again as they were, without the search. That is so only once they have come to rest:
/// none gliding to a new place, and none placed differently from the frame before (a
/// label that keeps its place does so by remembering where it was, and that memory must
/// have stopped changing too, or placing it afresh could still move it).
///
/// Where the labels would be placed, a frame calls `begin` and fills in the inputs it
/// returns (the bits of every number the picture and its labels are made from: the
/// frame's own, the camera and the canvas, are put in by `begin`), and then `settle`,
/// which says whether they are to be written again. If so, `replay` writes them; if
/// not, the scene places them as ever, through the writing methods here, which take
/// note of what they wrote, and then calls `end`, saying whether anything of its own
/// that the labels remember (where a name was, a glide) changed or is under way.
#[derive(Default)]
pub(super) struct Still {
    inputs: Inputs,
    last: Inputs,
    /// What the labels wrote when last placed, and what they wrote the time before.
    took: Vec<Took>,
    before: Vec<Took>,
    /// Whether the labels had come to rest when last placed; whether this frame's
    /// could not be taken down; whether this frame writes them again.
    rested: bool,
    spoilt: bool,
    again: bool,
}

impl Still {
    /// A new frame: the inputs to fill in, the frame's own already there.
    pub(super) fn begin(&mut self, f: &Frame) -> &mut Inputs {
        self.inputs.0.clear();
        self.inputs.frame(f);
        &mut self.inputs
    }

    /// The inputs being filled in, to add more.
    pub(super) fn inputs(&mut self) -> &mut Inputs {
        &mut self.inputs
    }

    /// Once the inputs are in: are the last labels to be written again? (Not if the
    /// frame no longer remembers where those that keep their place were: they were not
    /// written in the last frame, say while another scene was shown.)
    pub(super) fn settle(&mut self, f: &Frame) -> bool {
        let remembered = |t: &Took| match *t {
            Took::Placed { name, number, offset, .. } => f.places.chosen(name, number) == Some(offset),
            _ => true,
        };
        self.again = keeps() && self.rested && self.inputs == self.last && self.took.iter().all(remembered);
        #[cfg(test)]
        tried::AGAIN.set(tried::AGAIN.get() + self.again as u32);
        std::mem::swap(&mut self.inputs, &mut self.last);
        if !self.again {
            std::mem::swap(&mut self.took, &mut self.before);
            self.took.clear();
            (self.rested, self.spoilt) = (false, false);
        }
        self.again
    }

    /// The labels have been placed afresh: they have come to rest if nothing of the
    /// scene's own that they remember has `moved` or is still on its way somewhere (a
    /// glide that did not move only because no time passed is not over), none of those
    /// that keep their place is gliding, and they wrote what they wrote the time before.
    pub(super) fn end(&mut self, f: &Frame, moved: bool) {
        if !self.again {
            self.rested = !moved && !self.spoilt && !f.places.gliding() && self.took == self.before;
        }
    }

    /// Write the labels again as they were (and keep the labels placed after them out of
    /// the same boxes in `room`).
    pub(super) fn replay(&self, f: &mut Frame, room: &mut Room) {
        for took in &self.took {
            match took {
                Took::Text { pos, text, size, color, align, backed } => {
                    if *backed {
                        f.caption_px(*pos, text.as_str(), *size, *color, *align);
                    } else {
                        f.text_px(*pos, text.as_str(), *size, *color, *align);
                    }
                }
                Took::Placed { name, number, from, offset, text, size, color } => f.text_placed(name, *number, *from, *offset, text.as_str(), *size, *color),
                Took::Line(a, b, width, color) => hud_line(f, *a, *b, *width, *color),
                Took::Boxed(min, max) => room.boxed(*min, *max),
            }
        }
    }

    /// The boxes the labels last written keep the labels after them out of (as the bits
    /// of their corners), for the inputs of those.
    pub(super) fn boxes(&self, k: &mut Inputs) {
        for took in &self.took {
            if let Took::Boxed(min, max) = took {
                k.f32s(&[min.x, min.y, max.x, max.y]);
            }
        }
    }

    /// Take down the labels written since there were `mark` of them. `placed`: the name
    /// of one that keeps its place, and the point it is placed from, if it was written
    /// with `Frame::text_placed`. (If what was written is not what that would write, it
    /// cannot be written again from what is taken down: the labels are not kept.)
    fn took_since(&mut self, f: &Frame, mark: usize, placed: Option<(Tag, Vec2)>) {
        let new = f.dl.labels.get(mark..).unwrap_or(&[]);
        match (placed, new) {
            (Some((Tag(name, number), from)), [l]) => {
                let offset = f.places.chosen(name, number);
                let fits = offset.is_some_and(|o| from + o == l.pos) && l.align == Align2::CENTER_CENTER && !l.backed;
                if !fits {
                    self.spoilt = true;
                    return;
                }
                let offset = offset.unwrap_or(Vec2::ZERO);
                self.took.push(Took::Placed { name, number, from, offset, text: l.text.clone(), size: l.size, color: l.color });
            }
            (Some(_), _) if !new.is_empty() => self.spoilt = true,
            _ => {
                for l in new {
                    self.took.push(Took::Text { pos: l.pos, text: l.text.clone(), size: l.size, color: l.color, align: l.align, backed: l.backed });
                }
            }
        }
    }

    /// `Room::label_any`, taken down.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn label_any(&mut self, room: &mut Room, f: &mut Frame, name: impl Into<Tag>, points: &[Vec3], text: &str, size: f32, color: Color, towards: Vec2, gap: f32, optional: bool) -> bool {
        let (tag, mark) = (name.into(), f.dl.labels.len());
        // (a label is placed from its first point, if that is on the screen)
        let from = points.first().and_then(|p| on_screen(f, *p));
        let shown = room.label_any(f, tag, points, text, size, color, towards, gap, optional);
        self.took_since(f, mark, from.map(|from| (tag, from)));
        shown
    }

    /// `Room::label`, taken down.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn label(&mut self, room: &mut Room, f: &mut Frame, name: impl Into<Tag>, at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, gap: f32, optional: bool) -> bool {
        self.label_any(room, f, name, &[at], text, size, color, towards, gap, optional)
    }

    /// `Frame::text_placed`, taken down.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn text_placed(&mut self, f: &mut Frame, name: &'static str, number: u32, from: Vec2, offset: Vec2, text: impl Into<String>, size: f32, color: Color) {
        let mark = f.dl.labels.len();
        f.text_placed(name, number, from, offset, text, size, color);
        self.took_since(f, mark, Some((Tag(name, number), from)));
    }

    /// `Frame::text_px`, taken down.
    pub(super) fn text_px(&mut self, f: &mut Frame, pos: Vec2, text: impl Into<String>, size: f32, color: Color, align: Align2) {
        let mark = f.dl.labels.len();
        f.text_px(pos, text, size, color, align);
        self.took_since(f, mark, None);
    }

    /// `Frame::caption_px`, taken down.
    pub(super) fn caption_px(&mut self, f: &mut Frame, pos: Vec2, text: impl Into<String>, size: f32, color: Color, align: Align2) {
        let mark = f.dl.labels.len();
        f.caption_px(pos, text, size, color, align);
        self.took_since(f, mark, None);
    }

    /// `Room::boxed`, taken down.
    pub(super) fn boxed(&mut self, room: &mut Room, min: Vec2, max: Vec2) {
        room.boxed(min, max);
        self.took.push(Took::Boxed(min, max));
    }

    /// `hud_line`, taken down.
    pub(super) fn hud_line(&mut self, f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
        hud_line(f, a, b, width, color);
        self.took.push(Took::Line(a, b, width, color));
    }
}

/// Whether what is worked out is kept from frame to frame (the names, and words and
/// numbers kept with what they were made from). Always, but in the copy of the
/// application that the tests compare with, which works out everything afresh in every
/// frame (see `tried`).
#[cfg(not(test))]
pub(super) fn keeps() -> bool {
    true
}

#[cfg(test)]
pub(super) fn keeps() -> bool {
    !tried::NEVER.get()
}

/// For the tests of the scenes that keep their labels (see `Still`): the application run
/// twice side by side on the same input, one keeping its labels while it can and the
/// other placing them afresh in every frame, and what they draw compared after every
/// frame.
#[cfg(test)]
pub(super) mod tried {
    use crate::headless::{input, Stage};
    use std::cell::Cell;

    thread_local! {
        /// Place every label afresh in every frame (the copy that keeps nothing).
        pub(super) static NEVER: Cell<bool> = const { Cell::new(false) };
        /// How many times labels were written again as they were.
        pub(super) static AGAIN: Cell<u32> = const { Cell::new(0) };
    }

    /// One drawing device at a time (the software renderer is shared).
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// One thing done to the application.
    #[derive(Clone, Copy, Debug)]
    pub(in crate::scenes::linalg) enum Step {
        /// So many frames with nothing done.
        Rest(u32),
        /// The `k`-th thing to try pressed.
        Try(usize),
        /// A press at one point, a drag over ten frames to another, and a release
        /// (logical pixels): a handle, or the view turned, or a control in the panel.
        Drag([f32; 2], [f32; 2]),
        /// A click.
        Click([f32; 2]),
        /// The wheel turned at a point: the view zoomed.
        Scroll([f32; 2], f32),
        /// The window made this size, from now on.
        Size([f32; 2]),
        /// The scene's time stopped or started again (the space bar).
        Pause,
        /// Another scene shown for a few frames, and then this one again.
        Away,
    }

    /// Run `scene` through `steps` in both copies; panics at the first frame in which
    /// they draw anything differently. Returns how many times the labels were written
    /// again as they were (None where nothing can draw: then nothing was checked).
    pub(in crate::scenes::linalg) fn labels_follow(scene: &str, steps: &[Step]) -> Option<u32> {
        let _alone = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let mut stages = match (Stage::showing(scene), Stage::showing(scene)) {
            (Ok(keep), Ok(fresh)) => [keep, fresh],
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("SKIPPED: nothing here can draw ({e})");
                return None;
            }
        };
        for stage in &mut stages {
            stage.app.set_dry(true);
        }
        AGAIN.set(0);
        let (mut size, mut frame) = ([1600.0f32, 1000.0], 0u64);
        let mut run = |stages: &mut [Stage; 2], size: [f32; 2], events: Vec<egui::Event>, press: Option<usize>| {
            let mut shapes = Vec::new();
            for (stage, never) in stages.iter_mut().zip([false, true]) {
                NEVER.set(never);
                crate::ui::collect_panel(true);
                if let Some(k) = press {
                    crate::ui::press_experiment(k);
                }
                let out = stage.frame(input(size, frame, events.clone()));
                stage.done(&out.textures_delta);
                crate::ui::collect_panel(false);
                shapes.push(out.shapes);
            }
            NEVER.set(false);
            let [keep, fresh] = &*stages;
            let bits = |stage: &Stage| -> Vec<Vec<[u32; 2]>> { stage.app.drawn_points().iter().map(|b| b.iter().map(|p| p.to_array().map(f32::to_bits)).collect()).collect() };
            assert!(bits(keep) == bits(fresh), "{scene}, frame {frame}: what is drawn differs");
            assert_eq!(keep.app.drawn_labels(), fresh.app.drawn_labels(), "{scene}, frame {frame}");
            assert!(shapes[0] == shapes[1], "{scene}, frame {frame}: the labels or the panel differ");
            frame += 1;
        };
        let button = |at: [f32; 2], pressed| egui::Event::PointerButton { pos: egui::pos2(at[0], at[1]), button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        let moved = |at: [f32; 2]| egui::Event::PointerMoved(egui::pos2(at[0], at[1]));
        for step in steps {
            match *step {
                Step::Rest(n) => (0..n).for_each(|_| run(&mut stages, size, Vec::new(), None)),
                Step::Try(k) => run(&mut stages, size, Vec::new(), Some(k)),
                Step::Drag(a, b) => {
                    run(&mut stages, size, vec![moved(a)], None);
                    run(&mut stages, size, vec![button(a, true)], None);
                    for i in 1..=10 {
                        let t = i as f32 / 10.0;
                        run(&mut stages, size, vec![moved([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])], None);
                    }
                    run(&mut stages, size, vec![button(b, false)], None);
                }
                Step::Click(at) => {
                    run(&mut stages, size, vec![moved(at)], None);
                    run(&mut stages, size, vec![button(at, true)], None);
                    run(&mut stages, size, vec![button(at, false)], None);
                }
                Step::Scroll(at, amount) => {
                    let wheel = egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, amount), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers: Default::default() };
                    run(&mut stages, size, vec![moved(at), wheel], None);
                }
                Step::Size(new) => size = new,
                Step::Pause => {
                    let space = egui::Event::Key { key: egui::Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() };
                    run(&mut stages, size, vec![space], None);
                }
                Step::Away => {
                    for stage in &mut stages {
                        stage.app.select("vectors");
                    }
                    (0..3).for_each(|_| run(&mut stages, size, Vec::new(), None));
                    for stage in &mut stages {
                        stage.app.select(scene);
                    }
                }
            }
        }
        Some(AGAIN.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, from a scene that has been played with
    /// first (so that the button must set everything itself).
    fn trial(k: usize) -> Subspaces {
        let mut s = Subspaces::create();
        s.set_matrix([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0], [2.0, 2.0, 2.0]], true);
        (s.x, s.show_bases, s.show_same, s.show_image) = (DVec3::new(-1.0, 0.2, 0.3), false, !TRIALS[k].same, !TRIALS[k].image);
        s.set_up(k);
        s
    }

    /// The scene's time run on at 60 frames a second until nothing moves by itself,
    /// calling `each` after every frame.
    fn play(s: &mut Subspaces, mut each: impl FnMut(&Subspaces)) {
        for _ in 0..2000 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            s.glide(1.0 / 60.0, false);
            each(s);
            if !s.animated() {
                return;
            }
        }
        panic!("still moving after 2000 frames");
    }

    /// A matrix as its entries, row by row.
    fn rows(sp: &Spaces) -> [DVec3; 3] {
        sp.a.map(DVec3::from_array)
    }

    fn cols(sp: &Spaces) -> [DVec3; 3] {
        let r = rows(sp);
        [DVec3::new(r[0].x, r[1].x, r[2].x), DVec3::new(r[0].y, r[1].y, r[2].y), DVec3::new(r[0].z, r[1].z, r[2].z)]
    }

    /// How far `v` is from the span of `basis`.
    fn off(v: DVec3, basis: &[DVec3]) -> f64 {
        (v - basis.iter().map(|b| *b * b.dot(v)).sum::<DVec3>()).length()
    }

    /// Random matrices of every rank, with entries in steps of 0.01 like those typed in.
    fn matrices() -> Vec<Matrix> {
        let mut rng = Rng::new(11);
        let mut out = vec![RANK3, RANK2, RANK1, RANK0, NEARLY];
        let entry = |rng: &mut Rng| (rng.normal() * 0.6 * 100.0).round() / 100.0;
        for k in 0..300 {
            let mut a = [[0.0f32; 3]; 3];
            for row in a.iter_mut() {
                for v in row.iter_mut() {
                    *v = entry(&mut rng);
                }
            }
            // a third of rank 2 (a row the sum of the others) and some of rank 1
            if k % 3 == 1 {
                a[2] = [a[0][0] + a[1][0], a[0][1] + a[1][1], a[0][2] + a[1][2]];
            } else if k % 3 == 2 {
                let c = [entry(&mut rng), entry(&mut rng), entry(&mut rng)];
                a = [a[0].map(|v| v * c[0]), a[0].map(|v| v * c[1]), a[0].map(|v| v * c[2])];
            }
            out.push(a);
        }
        out
    }

    /// The four subspaces found are the ones their names say: the rows of A lie in
    /// the row space and the null space is at right angles to every row (A sends it
    /// to 0); the columns lie in the column space and the left null space is at right
    /// angles to every column; each basis is at right angles and of length 1; A vₖ =
    /// σₖ uₖ; and the transpose has the same rank (row rank = column rank).
    #[test]
    fn the_four_subspaces_are_what_their_names_say() {
        let mut seen = [0usize; 4];
        for a in matrices() {
            let sp = Spaces::of(&a);
            let r = sp.rank;
            seen[r] += 1;
            let size = sp.s[0].max(1e-12);
            for (bases, name) in [(&sp.v, "v"), (&sp.u, "u")] {
                for i in 0..3 {
                    for j in 0..3 {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!((bases[i].dot(bases[j]) - want).abs() < 1e-9, "{a:?}: {name}{i} . {name}{j}");
                    }
                }
            }
            let (row, null) = sp.v.split_at(r);
            let (col, left) = sp.u.split_at(r);
            // (to within what counts as zero: a matrix whose entries are rounded is
            // only nearly of a lower rank, by up to a millionth of its largest σ)
            let zero = 2.0 * TOL * size;
            for row_vector in rows(&sp) {
                assert!(off(row_vector, row) < zero, "{a:?}: a row is off the row space");
                assert!(null.iter().all(|n| n.dot(row_vector).abs() < zero), "{a:?}: the null space is not at right angles to a row");
            }
            for column in cols(&sp) {
                assert!(off(column, col) < zero, "{a:?}: a column is off the column space");
                assert!(left.iter().all(|w| w.dot(column).abs() < zero), "{a:?}: the left null space is not at right angles to a column");
            }
            for k in 0..r {
                assert!((sp.apply(sp.v[k]) - sp.u[k] * sp.s[k]).length() < 1e-9 * size, "{a:?}: A v{k} is not σ{k} u{k}");
            }
            for n in null {
                assert!(sp.apply(*n).length() <= 1e-6 * size, "{a:?}: A n = {}", sp.apply(*n));
            }
            let t = [0, 1, 2].map(|j| [0, 1, 2].map(|i| a[i][j]));
            assert_eq!(Spaces::of(&t).rank, r, "{a:?}: the transpose has another rank");
        }
        assert!(seen.iter().all(|&n| n > 0), "every rank seen: {seen:?}");
        let ranks = [RANK3, RANK2, RANK1, RANK0].map(|a| Spaces::of(&a).rank);
        assert_eq!(ranks, [3, 2, 1, 0]);
        assert!(PRESETS.iter().all(|(name, a)| *name == format!("rank {}", Spaces::of(a).rank)));
    }

    /// Every x is xᵣ + xₙ in exactly one way, at right angles; Ax = A xᵣ; and the
    /// part in the row space is the shortest input with that output.
    #[test]
    fn every_input_splits_into_a_row_part_and_a_null_part() {
        let mut rng = Rng::new(3);
        for a in matrices() {
            let sp = Spaces::of(&a);
            let size = sp.s[0].max(1e-12);
            for _ in 0..5 {
                let x = DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64);
                let (xr, xn) = sp.split(x);
                assert!((xr + xn - x).length() < 1e-12, "{a:?}");
                assert!(xr.dot(xn).abs() < 1e-9 * x.length_squared(), "{a:?}: the parts are not at right angles");
                assert!(off(xr, &sp.v[..sp.rank]) < 1e-9 && off(xn, &sp.v[sp.rank..]) < 1e-9, "{a:?}");
                assert!((sp.apply(x) - sp.apply(xr)).length() < 1e-6 * size * x.length(), "{a:?}: Ax is not A xr");
                assert!(xr.length() <= x.length() + 1e-12);
                // and the row part is the one input of the row space with that output
                if sp.rank > 0 {
                    assert!((sp.back(sp.apply(x)) - xr).length() < 1e-6 * x.length() * sp.s[0] / sp.s[sp.rank - 1], "{a:?}");
                } else {
                    assert_eq!(sp.back(sp.apply(x)), DVec3::ZERO);
                }
            }
        }
    }

    /// "Slide x along the null space": x runs along the dashed line, out to near both
    /// of its ends and back, its row part and Ax staying exactly where they are; the
    /// dashed line is every input with that output.
    #[test]
    fn sliding_x_along_the_null_space_leaves_ax_where_it_is() {
        // (every button sets the matrix, x and what is shown, whatever came before)
        for (k, t) in TRIALS.iter().enumerate() {
            let s = trial(k);
            assert!(s.a == t.a && s.show_bases && s.show_same == t.same && s.show_image == t.image && s.tour.is_some() == (t.moves != Moves::Still), "trial {k}");
            assert!((s.x - DVec3::from_array(t.x)).length() < 1e-12, "trial {k} starts x at {}", s.x);
        }
        let mut s = trial(0);
        assert!(s.spaces.rank == 2 && s.show_same && s.tour.is_some());
        let start = s.x;
        let (xr, _) = s.spaces.split(start);
        let n = s.spaces.v[2];
        let ax = s.spaces.apply(start);
        let reach = (R * R - xr.length_squared()).sqrt();
        let (mut low, mut high, mut frames) = (0.0f64, 0.0f64, 0);
        play(&mut s, |s| {
            let (now_r, now_n) = s.spaces.split(s.x);
            assert!((now_r - xr).length() < 1e-12 && (s.spaces.apply(s.x) - ax).length() < 1e-12, "xr or Ax moved");
            assert!(s.x.length() <= R && (now_n - n * n.dot(now_n)).length() < 1e-12, "x is off the dashed line");
            let along = n.dot(s.x - xr);
            (low, high) = (low.min(along), high.max(along));
            // the caption under the picture says where Ax is, and so does the panel
            assert_eq!(coords(s.spaces.apply(s.x)), "(0.63, 0.45, 0.18)");
            frames += 1;
        });
        assert!(low < -0.85 * reach && high > 0.85 * reach, "{low} {high} of {reach}");
        assert!((s.x - start).length() < 1e-9 && frames as f32 / 60.0 > TOUR_TIME * 0.95, "{frames} frames");
        // off the line, in the row space's directions, the output is another
        let v = s.spaces.v[0];
        assert!((s.spaces.apply(start + v * 0.1) - ax).length() > 0.05);
        // and the line is every input with that output: whatever is added along n
        for t in [-2.0, -0.3, 0.7, 5.0] {
            assert!((s.spaces.apply(start + n * t) - ax).length() < 1e-12);
        }
    }

    /// "Go round the row space": xᵣ goes once round a circle in the row space, Ax
    /// round an ellipse in the column space, and no two points of the circle land on
    /// one point: A never shortens a step within the row space below σ₂ times itself.
    /// Every point of the column space comes from one point of the row space.
    #[test]
    fn going_round_the_row_space_goes_round_the_column_space() {
        let mut s = trial(1);
        let (xr0, xn0) = s.spaces.split(s.x);
        let mut path: Vec<(DVec3, DVec3)> = vec![(xr0, s.spaces.apply(s.x))];
        play(&mut s, |s| {
            let (xr, xn) = s.spaces.split(s.x);
            assert!((xn - xn0).length() < 1e-12 && (xr.length() - xr0.length()).abs() < 1e-12 && s.x.length() <= R);
            let b = s.spaces.apply(s.x);
            assert!(s.spaces.u[2].dot(b).abs() < 1e-12, "Ax left the column space");
            path.push((xr, b));
        });
        // once round: the angle swept adds up to a whole turn
        let normal = s.spaces.v[2];
        let swept: f64 = path.windows(2).map(|w| w[0].0.cross(w[1].0).dot(normal).atan2(w[0].0.dot(w[1].0))).sum();
        assert!((swept.abs() - std::f64::consts::TAU).abs() < 1e-6, "{swept}");
        let sigma2 = s.spaces.s[1];
        for (i, a) in path.iter().enumerate().step_by(7) {
            for b in path.iter().skip(i + 1).step_by(5) {
                let (dx, db) = ((a.0 - b.0).length(), (a.1 - b.1).length());
                assert!(db >= sigma2 * dx * (1.0 - 1e-9), "two points of the circle land {db} apart from {dx}");
            }
        }
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let b = s.spaces.u[0] * rng.normal() as f64 + s.spaces.u[1] * rng.normal() as f64;
            let x = s.spaces.back(b);
            assert!((s.spaces.apply(x) - b).length() < 1e-9 && s.spaces.v[2].dot(x).abs() < 1e-12);
        }
        assert!(sigma2 > 0.7);
    }

    /// "Rank 1", "rank 3" and "nearly rank 2", as their captions say.
    #[test]
    fn the_other_experiments_show_what_their_captions_say() {
        // rank 1: the rows along (1, -1, -1), the columns along (1, 2, -1); a line and a
        // plane on each side; Ax on the green line wherever x is
        let s = trial(2);
        let sp = s.spaces;
        assert!(sp.rank == 1 && s.show_same);
        let (row, column) = (DVec3::new(1.0, -1.0, -1.0), DVec3::new(1.0, 2.0, -1.0));
        assert!(rows(&sp).iter().all(|r| r.cross(row).length() < 1e-7) && cols(&sp).iter().all(|c| c.cross(column).length() < 1e-7));
        assert!(card(&sp, 0, false, 1e4)[0].0 == "row space: a line, dimension 1" && card(&sp, 0, false, 1e4)[2].0 == "null space: a plane, dimension 2");
        assert!(card(&sp, 1, false, 1e4)[0].0 == "column space: a line, dimension 1" && card(&sp, 1, false, 1e4)[2].0 == "left null space: a plane, dimension 2");
        assert!(card(&sp, 0, false, 1e4)[4].0.starts_with("$1 + 2 = 3$"));
        let mut rng = Rng::new(2);
        for _ in 0..100 {
            let x = held(DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64));
            assert!(sp.apply(x).cross(sp.u[0]).length() < 1e-12);
        }

        // rank 3: all of space on each side, only the origin left over; each output
        // from this input and no other
        let s = trial(3);
        let sp = s.spaces;
        assert!(sp.rank == 3 && !s.show_same);
        assert!(card(&sp, 0, false, 1e4)[0].0 == "row space: all of space, dimension 3" && card(&sp, 0, false, 1e4)[2].0 == "null space: the origin alone, dimension 0");
        assert!(card(&sp, 1, false, 1e4)[2].0 == "left null space: the origin alone, dimension 0" && card(&sp, 1, false, 1e4)[4].0.starts_with("$3 + 0 = 3$"));
        assert!((sp.back(sp.apply(s.x)) - s.x).length() < 1e-12 && sp.s[2] > 0.5);

        // nearly rank 2: the middle entry moved from 0 to 0.01; σ3 = 0.0026, rank 3;
        // the ball lands on a pancake 370 times thinner than it is wide
        let s = trial(4);
        let sp = s.spaces;
        let moved: Vec<(usize, usize)> = (0..9).map(|k| (k / 3, k % 3)).filter(|&(i, j)| NEARLY[i][j] != RANK2[i][j]).collect();
        assert!(moved == [(1, 1)] && RANK2[1][1] == 0.0 && NEARLY[1][1] == 0.01);
        assert!(sp.rank == 3 && s.show_image && sigma(&sp, 2) == "0.0026" && num(sp.s[2], 4) == "0.0026", "{:?}", sp.s);
        assert!(sp.s[2] > 1e3 * TOL * sp.s[0]);
        assert_eq!(format!("{:.0}", sp.s[0] / sp.s[2] / 10.0), "37");
        assert!(card(&sp, 0, false, 1e4)[2].0 == "null space: the origin alone, dimension 0");
    }

    /// Wherever the panel and the drags leave things, and whatever the matrix:
    /// nothing drawn or read out is not a number, and what the cards say has happened has.
    #[test]
    fn degenerate_cases_leave_everything_finite() {
        let mut cases = matrices();
        cases.extend([[[1e-30, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[100.0; 3]; 3], [[-100.0, 100.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 1e-6]], [[f32::NAN, 1.0, 0.0], [0.0, f32::INFINITY, 0.0], [0.0, 0.0, 1.0]]]);
        for a in cases {
            let sp = Spaces::of(&a);
            assert!(sp.s.iter().all(|v| v.is_finite()) && sp.v.iter().chain(&sp.u).all(|v| v.is_finite()), "{a:?}");
            assert!(sp.out_scale().is_finite() && sp.out_scale() > 0.0 && sp.out_scale() <= 1.0);
            for x in [DVec3::ZERO, DVec3::new(1.0, 0.5, -0.3), DVec3::new(0.0, 0.0, 1e-12)] {
                let mut s = Subspaces::create();
                s.set_matrix(a, true);
                s.x = x;
                let (xr, xn) = sp.split(x);
                assert!(xr.is_finite() && xn.is_finite() && sp.apply(x).is_finite() && sp.back(sp.apply(x)).is_finite(), "{a:?} {x}");
                for b in [DVec3::ZERO, DVec3::new(2.0, -1.0, 0.4), DVec3::splat(1e9)] {
                    let y = s.reach_for(b);
                    assert!(y.is_finite() && y.length() <= R + 1e-9, "{a:?} {b}: {y}");
                }
                for moves in [Moves::AlongNull, Moves::RoundRow] {
                    if let Some(tour) = s.tour_for(moves) {
                        for t in [0.0, 0.3, 0.77, 1.0] {
                            assert!(tour.at(t).is_finite() && tour.at(t).length() <= R + 1e-9, "{a:?} {x} {moves:?}");
                        }
                    }
                }
                for side in 0..2 {
                    for narrow in [false, true] {
                        for (line, size, _) in card(&sp, side, narrow, 300.0) {
                            assert!(size.is_finite() && !line.contains("NaN") && !line.contains('?') && !line.contains("inf"), "{a:?}: {line}");
                        }
                    }
                }
                assert!((0..3).all(|k| !sigma(&sp, k).contains('?')));
            }
        }
        // the zero matrix: nothing reached but 0, everything ignored
        let sp = Spaces::of(&RANK0);
        assert!(sp.rank == 0 && sp.apply(DVec3::ONE) == DVec3::ZERO && sp.split(DVec3::ONE) == (DVec3::ZERO, DVec3::ONE));
        assert!(card(&sp, 0, false, 1e4)[0].0 == "row space: the origin alone, dimension 0" && card(&sp, 1, false, 1e4)[2].0 == "left null space: all of space, dimension 3");
        // a point outside the ball is brought to its edge
        assert!((held(DVec3::new(3.0, 4.0, 0.0)) - DVec3::new(0.9, 1.2, 0.0)).length() < 1e-12 && held(DVec3::splat(f64::NAN)) == DVec3::ZERO);
    }

    /// Dragging Ax moves x's row part to the one input of the row space with that
    /// output and keeps its null part; an output off the column space is taken to
    /// the nearest one there is.
    #[test]
    fn dragging_ax_moves_the_row_part_only() {
        let mut s = Subspaces::create();
        let (_, xn) = s.spaces.split(s.x);
        let b = s.spaces.u[0] * 0.3 - s.spaces.u[1] * 0.2;
        s.x = s.reach_for(b);
        assert!((s.spaces.apply(s.x) - b).length() < 1e-12 && (s.spaces.split(s.x).1 - xn).length() < 1e-12);
        s.x = s.reach_for(b + s.spaces.u[2] * 0.4);
        assert!((s.spaces.apply(s.x) - b).length() < 1e-12);
    }

    /// A new matrix fades in over the old one and the points glide to their places;
    /// then everything is exactly where it is and nothing moves. Paused, a change is
    /// shown at once.
    #[test]
    fn a_change_comes_to_rest() {
        for hz in [60.0f32, 144.0] {
            let mut s = Subspaces::create();
            assert!(!s.animated());
            s.set_up(2);
            assert!(s.fading.is_some() && s.animated());
            let mut frames = 0;
            while s.animated() {
                s.advance(1.0 / hz, 1.0 / hz);
                s.glide(1.0 / hz, false);
                frames += 1;
                assert!(frames < 2000);
            }
            assert!(s.drawn == s.target() && s.fading.is_none());
            assert!((frames as f32 / hz) < FADE_TIME + 0.6, "{frames} frames at {hz} Hz");
        }
        let mut s = Subspaces::create();
        s.set_matrix(RANK1, true);
        s.x = DVec3::new(-0.4, 0.2, 0.1);
        s.advance(0.0, 1.0 / 60.0);
        s.glide(1.0 / 60.0, true);
        assert!(s.fading.is_none() && !s.animated());
        // dragging an entry (not typed) changes the picture without a fade, unless the rank changes
        let mut s = Subspaces::create();
        let mut a = RANK3;
        a[0][0] += 0.01;
        s.set_matrix(RANK3, true);
        s.fading = None;
        s.set_matrix(a, false);
        assert!(s.fading.is_none());
        s.set_matrix(RANK2, false);
        assert!(s.fading.is_some());
    }

    /// The balls, cards, matrix and arrow share the canvas without leaving it or
    /// covering one another, whatever its shape.
    #[test]
    fn the_canvas_is_shared_on_every_screen() {
        for (x, y) in [(1070.0f32, 1000.0f32), (868.0, 700.0), (1600.0, 1000.0), (390.0, 800.0), (360.0, 640.0), (600.0, 900.0), (844.0, 390.0), (640.0, 480.0), (2000.0, 1300.0), (240.0, 180.0), (70.0, 754.0), (299.0, 900.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            for inset in [0.0, 46.0] {
                let lay = Layout::of(view, inset);
                let everything = [lay.centre[0], lay.centre[1], lay.matrix, lay.arrow.0, lay.arrow.1, lay.card[0].0, lay.card[1].1];
                assert!(everything.iter().all(|p| p.is_finite()) && lay.radius > 0.0, "{x}x{y}: {lay:?}");
                let d = lay.distance(y);
                assert!(d.is_finite() && d > 0.0);
                // however small, the balls are on the canvas and apart
                for c in lay.centre {
                    assert!(c.x - lay.radius >= -0.01 && c.x + lay.radius <= x + 0.01 && c.y + lay.radius <= y + 0.01, "{x}x{y}: {lay:?}");
                }
                assert!(lay.centre[0].distance(lay.centre[1]) >= 2.0 * lay.radius, "{x}x{y}: the balls overlap");
                assert_eq!(lay.words, x >= 300.0 && y >= 300.0);
                if x < 300.0 || y < 400.0 {
                    continue;
                }
                for (i, c) in lay.centre.iter().enumerate() {
                    // the ball, its card and the matrix all on the canvas and clear of each other
                    assert!(c.x - lay.radius >= 0.0 && c.x + lay.radius <= x && c.y - lay.radius >= inset && c.y + lay.radius <= y, "{x}x{y}: ball {i}");
                    let (lo, hi) = lay.card[i];
                    assert!(lo.x >= 0.0 && hi.x <= x && hi.y <= y + 0.5 && lo.y >= c.y + lay.radius, "{x}x{y}: card {i} {lo} {hi}");
                    let mat = matrix_size(3, 3, MATRIX_WIDEST, MATRIX_TEXT);
                    let nearest = c.clamp(lay.matrix, lay.matrix + mat);
                    assert!(nearest.distance(*c) > lay.radius, "{x}x{y}: the matrix is over ball {i}");
                }
                assert!(lay.centre[0].distance(lay.centre[1]) > 2.0 * lay.radius, "{x}x{y}: the balls overlap");
            }
        }
    }

    /// "Where the inputs in the ball land": every Ax of an x in the ball is inside the
    /// ellipsoid whose half-axes the white ellipses are (R σₖ uₖ), and the ellipses'
    /// points are such outputs. "Drawn N times smaller": with the outputs drawn at that
    /// scale, every Ax of the ball is inside the outputs' ball.
    #[test]
    fn the_ball_of_inputs_lands_inside_its_ellipsoid() {
        let mut rng = Rng::new(8);
        let mut stretched = matrices();
        stretched.push([[3.0, 0.0, 1.0], [0.5, 2.0, 0.0], [0.0, 1.0, 4.0]]);
        for a in stretched {
            let sp = Spaces::of(&a);
            for _ in 0..40 {
                let x = held(DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64) * 0.8);
                let b = sp.apply(x);
                let inside: f64 = (0..sp.rank).map(|k| (sp.u[k].dot(b) / (R * sp.s[k])).powi(2)).sum();
                assert!(inside <= 1.0 + 1e-9, "{a:?}: {x} lands outside, at {inside}");
                assert!((b * sp.out_scale()).length() <= R * (1.0 + 1e-9), "{a:?}: Ax drawn outside the ball");
            }
            // a point of an ellipse, R (cos t σ₀u₀ + sin t σ₁u₁), is where R (cos t v₀ + sin t v₁) lands
            let t = 0.7f64;
            let x = (sp.v[0] * t.cos() + sp.v[1] * t.sin()) * R;
            let drawn = (sp.u[0] * (sp.s[0] * t.cos()) + sp.u[1] * (sp.s[1] * t.sin())) * R;
            // (to within what counts as zero: the ellipses are drawn flat where a σ is)
            let flat: DVec3 = (sp.rank..2).map(|k| sp.u[k] * (sp.s[k] * if k == 0 { t.cos() } else { t.sin() } * R)).sum();
            assert!((sp.apply(x) - (drawn - flat)).length() < 2.0 * TOL * sp.s[0] * R + 1e-12, "{a:?}");
        }
        assert_eq!(outputs_heading(Spaces::of(&[[3.0, 0.0, 1.0], [0.5, 2.0, 0.0], [0.0, 1.0, 4.0]]).out_scale()).split(", ").count(), 2);
        assert_eq!(outputs_heading(Spaces::of(&RANK2).out_scale()), "outputs $Ax$");
    }

    /// What is kept from frame to frame (the names, `Still`; the words over the
    /// picture, `Words`) follows every change: run side by side with a copy that works
    /// everything out afresh in every frame, the scene draws exactly the same in every
    /// frame, through drags of x, of Ax and of the view, a zoom, a number of the matrix
    /// dragged, every experiment, a pause, windows of other shapes (a phone's, and one
    /// too small for words), and the scene opened again.
    #[test]
    fn what_is_kept_follows_every_change() {
        use super::tried::{labels_follow, Step::*};
        let steps = [
            Rest(30),
            Drag([1161.0, 396.0], [1090.0, 320.0]),
            Rest(20),
            Drag([651.0, 460.0], [590.0, 340.0]),
            Rest(20),
            Drag([1000.0, 820.0], [1100.0, 780.0]),
            Rest(40),
            Scroll([800.0, 500.0], -200.0),
            Rest(30),
            Drag([1390.0, 955.0], [1440.0, 955.0]),
            Rest(40),
            Try(0),
            Rest(420),
            Try(1),
            Rest(200),
            Drag([651.0, 460.0], [560.0, 420.0]),
            Rest(40),
            Try(2),
            Rest(60),
            Try(3),
            Rest(60),
            Try(4),
            Rest(60),
            Pause,
            Try(0),
            Rest(20),
            Pause,
            Rest(30),
            Size([390.0, 800.0]),
            Rest(40),
            Drag([281.0, 207.0], [240.0, 150.0]),
            Rest(30),
            Size([290.0, 290.0]),
            Rest(20),
            Size([1600.0, 1000.0]),
            Rest(30),
            Away,
            Rest(30),
        ];
        if let Some(again) = labels_follow("subspaces", &steps) {
            assert!(again > 500, "the names were written again as they were only {again} times");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (name, a) in PRESETS.iter().copied().chain([("nearly", NEARLY)]) {
            let sp = Spaces::of(&a);
            println!("{name}: rank {} s {:?} s1/s3 {}", sp.rank, sp.s, sp.s[0] / sp.s[2]);
        }
        for k in 0..TRIALS.len() {
            let s = trial(k);
            let (xr, xn) = s.spaces.split(s.x);
            println!("trial {k}: x {:?} xr {:?} |xr| {} xn {:?} |xn| {} Ax {:?}", s.x, xr, xr.length(), xn, xn.length(), s.spaces.apply(s.x));
        }
    }
}
