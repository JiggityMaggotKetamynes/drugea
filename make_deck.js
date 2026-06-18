/**
 * DrugEA slide deck — PptxGenJS
 * 11 slides covering: problem, architecture, representation,
 * EA algorithms, surrogate, main results, RQ1, RQ2, REVAC, summary.
 */

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title  = "DrugEA: Evolutionary Drug Discovery";

// ── Palette ──────────────────────────────────────────────────
const DARK  = "0F2340";
const NAVY  = "1C3F60";
const TEAL  = "0D9488";
const TEAL2 = "5EEAD4";
const BLUE  = "2563EB";
const CARD  = "EBF5FB";
const CARD2 = "F0FDF9";
const WHITE = "FFFFFF";
const LGRAY = "F1F5F9";
const MUTED = "64748B";
const GREEN = "059669";
const AMBER = "B45309";
const RED   = "DC2626";

// Shadow factory — never reuse the same object (pptxgenjs mutates in place)
const sh = () => ({ type:"outer", color:"000000", blur:5, offset:2, angle:135, opacity:0.10 });

// Image paths
const IMG = {
  cal:   "/Users/davisalley/drugea/surrogate/calibration.png",        // 900×900  (1:1)
  revac: "/Users/davisalley/drugea/surrogate/revac_convergence.png",  // 1350×600 (2.25:1)
  rq1:   "/Users/davisalley/drugea/surrogate/rq1_sigma_adaptation.png", // 1800×750 (2.4:1)
  rq2:   "/Users/davisalley/drugea/surrogate/rq2_weight_adaptation.png",// 1050×600 (1.75:1)
};

// ── Shared helpers ────────────────────────────────────────────
function slideHeader(s, txt) {
  s.addShape(pres.shapes.RECTANGLE, {
    x:0.35, y:0.1, w:9.3, h:0.6,
    fill:{color:DARK}, line:{color:DARK},
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x:0.35, y:0.1, w:0.08, h:0.6,
    fill:{color:TEAL}, line:{color:TEAL},
  });
  s.addText(txt, {
    x:0.55, y:0.1, w:9.0, h:0.6,
    fontSize:21, bold:true, color:WHITE, fontFace:"Calibri",
    valign:"middle", align:"left", margin:0,
  });
}

function card(s, x, y, w, h, opts) {
  const { fill=CARD, radius=false } = opts || {};
  s.addShape(radius ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, {
    x, y, w, h, fill:{color:fill}, line:{color:fill},
    shadow: sh(), ...(radius ? {rectRadius:0.08} : {}),
  });
}

function tealDot(s, x, y) {
  s.addShape(pres.shapes.OVAL, { x, y, w:0.12, h:0.12, fill:{color:TEAL}, line:{color:TEAL} });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK };

  // Left accent stripe
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:0.09, h:5.625, fill:{color:TEAL}, line:{color:TEAL} });

  // Main title
  s.addText("DrugEA", {
    x:0.35, y:0.55, w:6.5, h:1.55,
    fontSize:82, bold:true, color:WHITE, fontFace:"Calibri", align:"left", margin:0,
  });

  // Subtitle
  s.addText("Evolutionary Search for Drug-Like EGFR Inhibitors", {
    x:0.35, y:2.2, w:7.0, h:0.65,
    fontSize:19, color:TEAL2, fontFace:"Calibri", align:"left", margin:0,
  });

  // Thin divider
  s.addShape(pres.shapes.RECTANGLE, { x:0.35, y:3.0, w:3.2, h:0.03, fill:{color:TEAL}, line:{color:TEAL} });

  // Course info
  s.addText("Evolutionary Computing  ·  Prof. Francesca  ·  2026", {
    x:0.35, y:3.15, w:6.5, h:0.35, fontSize:12, color:"7AAFCA",
    fontFace:"Calibri", align:"left", margin:0,
  });

  // Three bottom stat boxes
  const stats = [
    { n:"10⁶⁰", l:"molecules in search space" },
    { n:"6",    l:"pharmacological constraints" },
    { n:"100%", l:"chemical validity (SELFIES)" },
  ];
  stats.forEach((st, i) => {
    const x = 0.35 + i * 3.05;
    s.addShape(pres.shapes.RECTANGLE, { x, y:4.35, w:2.85, h:1.0, fill:{color:"162B40"}, line:{color:"253F5A"} });
    s.addText(st.n, { x, y:4.4,  w:2.85, h:0.5,  fontSize:28, bold:true, color:TEAL, fontFace:"Calibri", align:"center", margin:0 });
    s.addText(st.l, { x, y:4.9,  w:2.85, h:0.3,  fontSize:9, color:"7AAFCA", fontFace:"Calibri", align:"center", margin:0 });
  });

  // Decorative circles (molecular feel)
  s.addShape(pres.shapes.OVAL, { x:7.9, y:0.3, w:1.9, h:1.9, fill:{color:"162B40"}, line:{color:"2E6BA0", width:2.5} });
  s.addShape(pres.shapes.OVAL, { x:8.5, y:2.5, w:1.3, h:1.3, fill:{color:"0A3F3A"}, line:{color:TEAL, width:2} });
  s.addShape(pres.shapes.LINE, { x:8.85, y:2.19, w:0, h:0.32, line:{color:"2E6BA0", width:2} });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 2 — Problem & Approach
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Problem & Approach");

  // Left column — problem
  card(s, 0.35, 0.88, 4.55, 2.0);
  s.addText("The Problem", { x:0.55, y:0.93, w:4.1, h:0.35, fontSize:13, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });
  s.addText([
    { text: "Target: ", options:{bold:true} }, { text:"EGFR kinase — overexpressed in lung & colorectal cancers\n", options:{} },
    { text: "Scale: ", options:{bold:true} }, { text:"Chemical space ≈ 10⁶⁰ drug-like molecules\n", options:{} },
    { text: "Cost: ", options:{bold:true} }, { text:"Wet-lab screening: $2.6B and 12 years per drug\n", options:{} },
    { text: "Goal: ", options:{bold:true} }, { text:"Find feasible, tightly-binding molecules computationally", options:{} },
  ], {
    x:0.55, y:1.32, w:4.15, h:1.4,
    fontSize:12, color:NAVY, fontFace:"Calibri", align:"left", valign:"top",
  });

  // Left column — CSP formulation
  card(s, 0.35, 3.02, 4.55, 2.3);
  s.addText("CSP Formulation  (Lecture 13)", { x:0.55, y:3.07, w:4.1, h:0.35, fontSize:13, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });
  const csp = [
    ["S", "Search space of fragment-assembled molecules  (~10⁶⁰)"],
    ["C", "6 hard constraints: MW, logP, HBD, HBA, SA, PAINS"],
    ["N", "Neighborhood: fragment substitution / insertion / deletion"],
  ];
  csp.forEach(([sym, desc], i) => {
    const y = 3.48 + i * 0.55;
    s.addShape(pres.shapes.OVAL, { x:0.55, y, w:0.28, h:0.28, fill:{color:TEAL}, line:{color:TEAL} });
    s.addText(sym, { x:0.55, y, w:0.28, h:0.28, fontSize:11, bold:true, color:WHITE, fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    s.addText(desc, { x:0.92, y:y+0.01, w:3.85, h:0.28, fontSize:11, color:NAVY, fontFace:"Calibri", align:"left", valign:"middle", margin:0 });
  });

  // Right column — Our approach
  card(s, 5.05, 0.88, 4.55, 2.0);
  s.addText("Our Approach", { x:5.25, y:0.93, w:4.1, h:0.35, fontSize:13, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });
  s.addText([
    { text:"An Evolutionary Algorithm ", options:{bold:true} },
    { text:"searches the chemical space.\n\nA PyTorch MLP (AffinityMLP) trained on ", options:{} },
    { text:"10,721 EGFR molecules from ChEMBL203 ", options:{bold:true} },
    { text:"predicts binding free energy ΔG_bind, replacing expensive molecular docking.", options:{} },
  ], {
    x:5.25, y:1.32, w:4.2, h:1.4,
    fontSize:12, color:NAVY, fontFace:"Calibri",
  });

  // Fitness function box
  card(s, 5.05, 3.02, 4.55, 2.3, {fill:DARK});
  s.addText("Unified Fitness Function", { x:5.25, y:3.07, w:4.15, h:0.35, fontSize:13, bold:true, color:TEAL2, fontFace:"Calibri", margin:0 });
  s.addText("eval(x)  =  −ΔG_bind(x)  +  Σⱼ wⱼ · vⱼ(x)", {
    x:5.15, y:3.53, w:4.35, h:0.55,
    fontSize:15, bold:true, color:WHITE, fontFace:"Calibri", align:"center", margin:0,
  });
  s.addText([
    { text:"−ΔG_bind ", options:{bold:true, color:TEAL2} }, { text:"predicted binding energy (minimize → tighter binding)\n", options:{color:"AACFE0"} },
    { text:"wⱼ · vⱼ(x) ", options:{bold:true, color:"F0A857"} }, { text:"penalty for violated pharmacological constraints\n", options:{color:"AACFE0"} },
    { text:"feasible ", options:{bold:true, color:GREEN} }, { text:"when penalty term = 0  (all 6 constraints satisfied)", options:{color:"AACFE0"} },
  ], {
    x:5.25, y:4.12, w:4.25, h:1.1, fontSize:10, fontFace:"Calibri",
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 3 — Code Architecture
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Code Architecture  —  6 Modules");

  const modules = [
    { file:"representation.py", role:"Chromosome · SELFIES decode · 39-token vocab",        color:"0369A1" },
    { file:"fitness.py",        role:"6 Lipinski constraints · AffinityMLP surrogate",       color:"0D9488" },
    { file:"operators.py",      role:"Mutation (4 ops) · Crossover (2 ops) · σ/w adaptation",color:"7C3AED" },
    { file:"selection.py",      role:"Fitness sharing · Tournament · (μ+λ) · Archive",       color:"B45309" },
    { file:"main.py",           role:"EA loop · CLI · per-gen logging",                      color:NAVY    },
    { file:"tuning.py",         role:"REVAC hyperparameter tuning (50 evaluations)",          color:"9D174D" },
  ];

  // Two rows of 3
  modules.forEach((m, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.35 + col * 3.12;
    const y = 0.88 + row * 2.32;

    s.addShape(pres.shapes.RECTANGLE, { x, y, w:3.0, h:2.1, fill:{color:LGRAY}, line:{color:"E2E8F0"}, shadow:sh() });
    // Colour top bar
    s.addShape(pres.shapes.RECTANGLE, { x, y, w:3.0, h:0.45, fill:{color:m.color}, line:{color:m.color} });
    s.addText(m.file, { x:x+0.08, y:y+0.03, w:2.85, h:0.4, fontSize:12, bold:true, color:WHITE, fontFace:"Calibri", align:"left", margin:0 });
    s.addText(m.role, { x:x+0.08, y:y+0.5, w:2.85, h:1.5, fontSize:11, color:NAVY, fontFace:"Calibri", align:"left", valign:"top" });

    // Arrows between boxes in row 1 (first 2 cols)
    if (col < 2) {
      s.addShape(pres.shapes.LINE, { x:x+3.0, y:y+1.0, w:0.12, h:0, line:{color:MUTED, width:1.2} });
      // arrowhead approximation
      s.addText("▶", { x:x+3.07, y:y+0.87, w:0.18, h:0.25, fontSize:8, color:MUTED, fontFace:"Calibri", align:"center", margin:0 });
    }
  });

  // surrogate label
  s.addShape(pres.shapes.RECTANGLE, {
    x:3.59, y:4.95, w:2.85, h:0.43,
    fill:{color:"EDE9FE"}, line:{color:"7C3AED", width:1},
  });
  s.addText("surrogate/model.py + train.py  →  loaded by fitness.py", {
    x:3.62, y:4.98, w:2.8, h:0.38, fontSize:9, color:"5B21B6", fontFace:"Calibri", align:"center", margin:0,
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 4 — Chromosome & Group SELFIES
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Molecular Representation  —  Group SELFIES");

  // SELFIES guarantee (top)
  card(s, 0.35, 0.88, 9.3, 1.0, {fill:CARD2});
  s.addShape(pres.shapes.RECTANGLE, { x:0.35, y:0.88, w:0.08, h:1.0, fill:{color:GREEN}, line:{color:GREEN} });
  s.addText("Why SELFIES?", { x:0.55, y:0.92, w:2.0, h:0.32, fontSize:13, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });
  s.addText([
    { text:"SMILES-based mutation: ", options:{bold:true, color:RED} },
    { text:"30–50% of offspring are chemically invalid  →  wasted evaluations\n", options:{color:NAVY} },
    { text:"Group SELFIES (Nigam et al., 2023): ", options:{bold:true, color:GREEN} },
    { text:"every possible token sequence decodes to a valid molecule by construction  →  ", options:{color:NAVY} },
    { text:"0% failure rate.  Tested: 200/200 random chromosomes valid.", options:{bold:true, color:GREEN} },
  ], { x:0.55, y:1.27, w:9.0, h:0.5, fontSize:11, fontFace:"Calibri" });

  // Chromosome structure diagram
  card(s, 0.35, 2.05, 5.6, 3.25);
  s.addText("Chromosome Structure", { x:0.55, y:2.1, w:5.1, h:0.35, fontSize:13, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });

  // Three gene sections
  const sections = [
    { label:"b₁ … bₖ", sub:"Fragment genes", desc:"Integer index into 39-token SELFIES vocabulary.\nControls which chemical building block occupies each position.", color:"0369A1" },
    { label:"σ₁ … σₖ", sub:"Step-size genes", desc:"Per-fragment mutation probability.\nSelf-adapted by EA — low σ = frozen core; high σ = exploring.", color:"7C3AED" },
    { label:"w₁ … w₆", sub:"Penalty weights", desc:"Per-constraint penalty coefficient.\nSelf-adapted — high w = EA learned this constraint is hard.", color:AMBER },
  ];

  sections.forEach((sec, i) => {
    const y = 2.6 + i * 0.82;
    s.addShape(pres.shapes.RECTANGLE, { x:0.55, y, w:1.35, h:0.6, fill:{color:sec.color}, line:{color:sec.color} });
    s.addText(sec.label, { x:0.55, y:y+0.01, w:1.35, h:0.3, fontSize:13, bold:true, color:WHITE, fontFace:"Calibri", align:"center", margin:0 });
    s.addText(sec.sub,   { x:0.55, y:y+0.32, w:1.35, h:0.27, fontSize:8,  color:WHITE, fontFace:"Calibri", align:"center", margin:0 });
    s.addText(sec.desc, { x:2.02, y:y+0.03, w:3.8, h:0.55, fontSize:10, color:NAVY, fontFace:"Calibri", valign:"middle" });
  });

  // Variable length note
  s.addText("K is variable length  (K_MIN=2, K_MAX=8)  —  insertion and deletion operators change chromosome length.", {
    x:0.55, y:5.07, w:5.3, h:0.32, fontSize:9, italic:true, color:MUTED, fontFace:"Calibri",
  });

  // Fragment vocabulary table (right)
  card(s, 6.1, 2.05, 3.55, 3.25);
  s.addText("Fragment Vocabulary  (|V| = 39)", { x:6.25, y:2.1, w:3.2, h:0.35, fontSize:12, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });

  const vocabGroups = [
    { cat:"Aromatic cores",   count:10, color:"0369A1", examples:"benzene, pyridine, indole, quinoline…" },
    { cat:"Ring systems",     count: 9, color:"7C3AED", examples:"piperidine, piperazine, morpholine…"    },
    { cat:"Linkers/pharma",   count: 8, color:"059669", examples:"amide, sulfonamide, urea, ester…"       },
    { cat:"R-group decorators", count:12, color:AMBER, examples:"methyl, methoxy, amine, halomethyl…"    },
  ];
  vocabGroups.forEach((g, i) => {
    const y = 2.57 + i * 0.67;
    s.addShape(pres.shapes.RECTANGLE, { x:6.25, y, w:0.35, h:0.5, fill:{color:g.color}, line:{color:g.color} });
    s.addText(String(g.count), { x:6.25, y, w:0.35, h:0.5, fontSize:14, bold:true, color:WHITE, fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    s.addText(g.cat,     { x:6.67, y:y+0.0,  w:2.85, h:0.25, fontSize:11, bold:true, color:NAVY,  fontFace:"Calibri", margin:0 });
    s.addText(g.examples,{ x:6.67, y:y+0.25, w:2.85, h:0.22, fontSize:8.5, color:MUTED, fontFace:"Calibri", margin:0 });
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 5 — Four EA Mechanisms
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "EA Algorithms  —  Four Course Mechanisms");

  const mechs = [
    {
      title:"Self-Adaptive Step Sizes σᵢ",
      ref:"Lecture 15",
      body:"Each chromosome carries its own mutation rates — one per fragment position.\n\nσᵢ′ = σᵢ · exp(τ′·N(0,1) + τ·Nᵢ(0,1))\n\nFragments whose mutation consistently hurts fitness evolve σᵢ → 0 (frozen). Peripheral positions keep σᵢ high (exploring).",
      color:"0369A1",
    },
    {
      title:"Self-Adaptive Penalty Weights wⱼ",
      ref:"Lecture 17",
      body:"Each chromosome carries its own penalty coefficients — one per constraint.\n\nwⱼ′ = wⱼ · exp(η·N(0,1))\n\nChromosomes that evolve higher weight on a hard constraint produce more feasible offspring and are selected more often, propagating high wⱼ.",
      color:AMBER,
    },
    {
      title:"Fitness Sharing over Tanimoto",
      ref:"Lecture 16",
      body:"F′(i) = F(i) · Σⱼ sh(d(i,j))\nsh(d) = 1 − (d/σ_share)^α  if d ≤ 0.3\n\nd(i,j) = 1 − Tanimoto(fp_i, fp_j)\n\nCrowded chemical regions get penalised → population forced to explore multiple distinct drug scaffolds simultaneously.",
      color:"7C3AED",
    },
    {
      title:"(μ+λ) Selection + Elitist Archive",
      ref:"Lecture 16",
      body:"Best μ survivors from μ parents + λ offspring.\nRank: feasible first, then by fitness.\n\nElitistArchive: stores all-time best feasible molecule. If a generation loses all feasible individuals, the archive champion is reinjected — best solution is never permanently lost.",
      color:"9D174D",
    },
  ];

  mechs.forEach((m, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.35 + col * 4.82;
    const y = 0.88  + row * 2.35;

    s.addShape(pres.shapes.RECTANGLE, { x, y, w:4.6, h:2.22, fill:{color:LGRAY}, line:{color:"E2E8F0"}, shadow:sh() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w:4.6, h:0.48, fill:{color:m.color}, line:{color:m.color} });
    s.addText(m.title, { x:x+0.1, y:y+0.04, w:3.5, h:0.4, fontSize:13, bold:true, color:WHITE, fontFace:"Calibri", align:"left", margin:0 });
    s.addText(m.ref,   { x:x+3.7, y:y+0.1,  w:0.8, h:0.3, fontSize:9, color:"DDE9FF", fontFace:"Calibri", align:"right", margin:0 });
    s.addText(m.body,  { x:x+0.1, y:y+0.55, w:4.38, h:1.55, fontSize:10.5, color:NAVY, fontFace:"Calibri", valign:"top" });
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 6 — Surrogate Model
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Surrogate Model  —  AffinityMLP");

  // Left: architecture
  card(s, 0.35, 0.88, 4.55, 4.45);
  s.addText("Architecture", { x:0.55, y:0.93, w:4.1, h:0.35, fontSize:13, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });

  const layers = [
    { label:"Morgan Fingerprint  (radius=2, 2048 bits)",  w:"full", color:MUTED   },
    { label:"Linear 2048 → 512  +  BatchNorm  +  ReLU",   w:"full", color:"0369A1" },
    { label:"Dropout(0.2)",                                w:"half", color:"7C3AED" },
    { label:"Linear 512 → 128  +  BatchNorm  +  ReLU",    w:"full", color:"0369A1" },
    { label:"Dropout(0.2)",                                w:"half", color:"7C3AED" },
    { label:"Linear 128 → 1",                             w:"third",color:TEAL    },
    { label:"ΔG_bind  (kcal/mol)",                        w:"third",color:GREEN   },
  ];
  layers.forEach((l, i) => {
    const bw = l.w === "full" ? 3.8 : l.w === "half" ? 2.0 : 1.4;
    const bx = 0.55 + (3.8 - bw) / 2;
    const by = 1.35 + i * 0.42;
    s.addShape(pres.shapes.RECTANGLE, { x:bx, y:by, w:bw, h:0.36, fill:{color:l.color}, line:{color:l.color} });
    s.addText(l.label, { x:bx, y:by, w:bw, h:0.36, fontSize:9, color:WHITE, fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    // arrow down (except last)
    if (i < layers.length - 1) {
      s.addText("↓", { x:bx + bw/2 - 0.1, y:by+0.35, w:0.2, h:0.09, fontSize:7, color:MUTED, fontFace:"Calibri", align:"center", margin:0 });
    }
  });

  s.addText("1,116,161 parameters  ·  MC Dropout: 50 passes → mean ± std", {
    x:0.45, y:4.42, w:4.35, h:0.55, fontSize:9, italic:true, color:MUTED, fontFace:"Calibri", align:"center",
  });

  // Right: calibration image
  s.addImage({ path:IMG.cal, x:5.1, y:0.88, w:4.5, h:4.5 });

  // Bottom stat bar (overlaid on right col)
  card(s, 5.1, 4.5, 4.5, 0.83, {fill:DARK});
  const metrics = [
    { n:"1.01", u:"kcal/mol", l:"RMSE  (<1.5 target ✓)" },
    { n:"0.83", u:"",         l:"Pearson R" },
    { n:"0.955",u:"",         l:"Spearman ρ (σ=0.5 noise)" },
  ];
  metrics.forEach((m, i) => {
    const x = 5.1 + i * 1.5;
    s.addText(m.n, { x, y:4.53, w:1.5, h:0.38, fontSize:18, bold:true, color:TEAL, fontFace:"Calibri", align:"center", margin:0 });
    s.addText(m.l, { x, y:4.92, w:1.5, h:0.3,  fontSize:8,  color:"AACFE0", fontFace:"Calibri", align:"center", margin:0 });
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 7 — Main EA Results
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "Main EA Results  —  100 Generations");

  // Archive convergence chart (left)
  const archiveGens = [0,1,2,3,5,7,9,13,14,16,32,46,60,66,72,73,91];
  const archiveFit  = [6.4549,6.4180,6.0118,5.5177,5.4235,5.2867,5.2380,5.1855,5.0685,4.1883,4.0263,3.8775,3.8104,3.6451,3.4520,3.3854,3.1848];

  s.addChart(pres.charts.LINE, [{
    name:"Archive best fitness",
    labels: archiveGens.map(String),
    values: archiveFit,
  }], {
    x:0.35, y:0.88, w:5.6, h:3.6,
    lineSize:3, lineSmooth:false,
    chartColors:["0D9488"],
    chartArea:{ fill:{ color:LGRAY }, roundedCorners:false },
    catAxisLabelColor:MUTED,
    valAxisLabelColor:MUTED,
    valGridLine:{ color:"E2E8F0", size:0.5 },
    catGridLine:{ style:"none" },
    showLegend:false,
    showTitle:true, title:"Archive Best Fitness vs. Generation",
    titleColor:NAVY, titleFontSize:12,
    catAxisTitle:"Generation", valAxisTitle:"Fitness (lower = better)",
  });

  // Stats below chart
  const cstats = [
    { n:"17", l:"archive improvements" },
    { n:"6.45", l:"gen 0 best fitness" },
    { n:"3.18", l:"gen 100 best fitness" },
  ];
  cstats.forEach((cs, i) => {
    const x = 0.35 + i * 1.9;
    s.addShape(pres.shapes.RECTANGLE, { x, y:4.6, w:1.8, h:0.8, fill:{color:DARK}, line:{color:DARK} });
    s.addText(cs.n, { x, y:4.64, w:1.8, h:0.38, fontSize:22, bold:true, color:TEAL, fontFace:"Calibri", align:"center", margin:0 });
    s.addText(cs.l, { x, y:5.01, w:1.8, h:0.28, fontSize:9,  color:"7AAFCA", fontFace:"Calibri", align:"center", margin:0 });
  });

  // Right: best molecule card
  card(s, 6.1, 0.88, 3.55, 2.55, {fill:DARK});
  s.addText("Best Molecule Found", { x:6.3, y:0.93, w:3.15, h:0.35, fontSize:12, bold:true, color:TEAL2, fontFace:"Calibri", margin:0 });
  const props = [
    ["Fitness",      "3.18"],
    ["ΔG_bind",      "−3.18 kcal/mol"],
    ["MW",           "489.6 Da"],
    ["logP",         "4.00"],
    ["HBD / HBA",    "3 / 10"],
    ["SA score",     "5.51"],
    ["PAINS alerts", "0"],
    ["Feasible",     "Yes ✓"],
  ];
  props.forEach(([k, v], i) => {
    const y = 1.35 + i * 0.255;
    s.addText(k,  { x:6.3,  y, w:1.5, h:0.23, fontSize:9.5, color:"7AAFCA", fontFace:"Calibri", margin:0 });
    s.addText(v,  { x:7.9,  y, w:1.6, h:0.23, fontSize:9.5, bold:true, color:k==="Feasible" ? TEAL2 : WHITE, fontFace:"Calibri", align:"right", margin:0 });
  });

  // Premature convergence resolved box
  card(s, 6.1, 3.57, 3.55, 1.78, {fill:CARD2});
  s.addShape(pres.shapes.RECTANGLE, { x:6.1, y:3.57, w:0.07, h:1.78, fill:{color:GREEN}, line:{color:GREEN} });
  s.addText("Premature Convergence: RESOLVED", { x:6.28, y:3.62, w:3.2, h:0.35, fontSize:11, bold:true, color:GREEN, fontFace:"Calibri", margin:0 });
  s.addText([
    { text:"Mock surrogate: ", options:{bold:true} }, { text:"~3 improvements, plateau at gen 30\n", options:{} },
    { text:"Real AffinityMLP: ", options:{bold:true} }, { text:"17 improvements, still improving at gen 100", options:{} },
  ], { x:6.28, y:4.04, w:3.25, h:1.2, fontSize:10.5, color:NAVY, fontFace:"Calibri" });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 8 — RQ1: σᵢ Adaptation
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "RQ1 — Do Self-Adapted σᵢ Recover the Pharmacophore?");

  // Data
  const frags = [
    { name:"imidazole",     cat:"ring",  sigma:0.026, n:45 },
    { name:"butyl",         cat:"rgrp",  sigma:0.035, n:30 },
    { name:"pyrazine",      cat:"core",  sigma:0.100, n:39 },
    { name:"methoxy",       cat:"rgrp",  sigma:0.112, n:30 },
    { name:"indole",        cat:"core",  sigma:0.148, n:22 },
    { name:"benzimidazole", cat:"core",  sigma:0.152, n: 9 },
    { name:"pyridine",      cat:"core",  sigma:0.177, n:15 },
    { name:"piperazine",    cat:"ring",  sigma:0.237, n: 8 },
    { name:"quinoline",     cat:"core",  sigma:0.759, n: 1 },
  ];
  const catColors = { core:"0369A1", ring:"7C3AED", rgrp:AMBER };
  const catLabels = { core:"aromatic core", ring:"ring system", rgrp:"R-group" };

  // Bar chart (native)
  s.addChart(pres.charts.BAR, [{
    name:"Mean σᵢ",
    labels: frags.map(f => f.name),
    values: frags.map(f => f.sigma),
  }], {
    x:0.35, y:0.88, w:6.0, h:3.55,
    barDir:"col",
    chartColors: frags.map(f => catColors[f.cat]),
    chartArea:{ fill:{ color:LGRAY } },
    catAxisLabelColor:MUTED, valAxisLabelColor:MUTED,
    valGridLine:{ color:"E2E8F0", size:0.5 }, catGridLine:{ style:"none" },
    showValue:true, dataLabelFontSize:9, dataLabelColor:NAVY,
    showLegend:false,
    showTitle:true, title:"Mean adapted σᵢ per fragment  (init = 0.50)",
    titleColor:NAVY, titleFontSize:11,
    valAxisMinVal:0, valAxisMaxVal:0.85,
  });

  // Key finding
  card(s, 0.35, 4.55, 6.0, 0.85, {fill:CARD2});
  s.addShape(pres.shapes.RECTANGLE, { x:0.35, y:4.55, w:0.07, h:0.85, fill:{color:GREEN}, line:{color:GREEN} });
  s.addText("Key Finding:", { x:0.55, y:4.58, w:1.1, h:0.3, fontSize:10, bold:true, color:GREEN, fontFace:"Calibri", margin:0 });
  s.addText("N-heterocycles (imidazole 0.026, pyrazine 0.10, pyridine 0.18) converged to near-zero σ — the EA independently discovered that nitrogen-rich aromatics are the EGFR pharmacophore. This matches medicinal chemistry consensus.", {
    x:1.7, y:4.58, w:4.5, h:0.78, fontSize:9.5, color:NAVY, fontFace:"Calibri",
  });

  // Legend + interpretation (right)
  card(s, 6.5, 0.88, 3.15, 3.55);
  s.addText("σᵢ Interpretation", { x:6.65, y:0.93, w:2.85, h:0.35, fontSize:12, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });

  const interps = [
    { range:"< 0.25", label:"Frozen", desc:"EA decided not to mutate this fragment — it's critical for binding.", color:GREEN },
    { range:"0.25–0.55", label:"Transitional", desc:"Moderate exploration — position partially converged.", color:AMBER },
    { range:"> 0.55", label:"Active", desc:"Still exploring — position not yet critical.", color:RED },
  ];
  interps.forEach((it, i) => {
    const y = 1.38 + i * 0.75;
    s.addShape(pres.shapes.RECTANGLE, { x:6.65, y, w:0.75, h:0.22, fill:{color:it.color}, line:{color:it.color} });
    s.addText(it.range, { x:6.65, y, w:0.75, h:0.22, fontSize:8.5, bold:true, color:WHITE, fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    s.addText(it.label, { x:7.47, y, w:2.0, h:0.22, fontSize:10, bold:true, color:it.color, fontFace:"Calibri", margin:0 });
    s.addText(it.desc, { x:6.65, y:y+0.25, w:2.85, h:0.45, fontSize:9.5, color:MUTED, fontFace:"Calibri" });
  });

  // Category legend
  s.addText("Fragment categories:", { x:6.65, y:3.68, w:2.85, h:0.28, fontSize:10, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });
  [["core","Aromatic core"],["ring","Ring system"],["rgrp","R-group"]].forEach(([k,l],i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:6.65, y:4.02+i*0.28, w:0.18, h:0.18, fill:{color:catColors[k]}, line:{color:catColors[k]} });
    s.addText(l, { x:6.9, y:4.01+i*0.28, w:2.5, h:0.2, fontSize:9.5, color:NAVY, fontFace:"Calibri", margin:0 });
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 9 — RQ2: Penalty Weight Evolution
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "RQ2 — Which Pharmacological Constraints Are Hardest?");

  const wdata = [
    { name:"MW",    w:1.4759, easy:false },
    { name:"logP",  w:1.7215, easy:false },
    { name:"HBD",   w:1.2493, easy:false },
    { name:"HBA",   w:1.1911, easy:false },
    { name:"SA",    w:0.7414, easy:true  },
    { name:"PAINS", w:1.4789, easy:false },
  ];

  // Bar chart
  s.addChart(pres.charts.BAR, [{
    name:"Mean evolved wⱼ",
    labels: wdata.map(d => d.name),
    values: wdata.map(d => d.w),
  }], {
    x:0.35, y:0.88, w:5.6, h:3.55,
    barDir:"col",
    chartColors: wdata.map(d => d.easy ? GREEN : (d.name==="logP" ? RED : "0369A1")),
    chartArea:{ fill:{ color:LGRAY } },
    catAxisLabelColor:MUTED, valAxisLabelColor:MUTED,
    valGridLine:{ color:"E2E8F0", size:0.5 }, catGridLine:{ style:"none" },
    showValue:true, dataLabelFontSize:10, dataLabelColor:NAVY,
    showLegend:false,
    showTitle:true, title:"Mean evolved penalty weight wⱼ  (init = 1.00)",
    titleColor:NAVY, titleFontSize:11,
    valAxisMinVal:0, valAxisMaxVal:2.1,
  });

  // Key finding
  card(s, 0.35, 4.55, 5.6, 0.85, {fill:CARD2});
  s.addShape(pres.shapes.RECTANGLE, { x:0.35, y:4.55, w:0.07, h:0.85, fill:{color:TEAL}, line:{color:TEAL} });
  s.addText("logP ranked #1 hardest (1.72).  PAINS #2, MW #3.  SA is easiest (0.74 — dropped below init because fragment assembly rarely violates synthetic accessibility).  EA correctly inferred difficulty from first principles.", {
    x:0.55, y:4.6, w:5.3, h:0.75, fontSize:10, color:NAVY, fontFace:"Calibri",
  });

  // Right: interpretation cards
  const interps = [
    { rank:1, name:"logP", w:"1.72", color:RED, why:"Aromatic fragment stacking pushes lipophilicity past the logP ≤ 5 limit. The EA learned to punish it hardest." },
    { rank:2, name:"PAINS", w:"1.48", color:AMBER, why:"Reactive electrophiles and structural alerts appear in certain fragment combinations. EA evolved resistance." },
    { rank:3, name:"MW", w:"1.48", color:"0369A1", why:"More fragments = better binding but higher MW. EA enforces balance." },
    { rank:6, name:"SA", w:"0.74", color:GREEN, why:"Fragment SELFIES builds from pre-synthesizable units — SA constraint almost never violated." },
  ];
  interps.forEach((it, i) => {
    const y = 0.88 + i * 1.17;
    card(s, 6.1, y, 3.55, 1.07);
    s.addShape(pres.shapes.RECTANGLE, { x:6.1, y, w:0.5, h:1.07, fill:{color:it.color}, line:{color:it.color} });
    s.addText(`#${it.rank}`, { x:6.1, y:y+0.02, w:0.5, h:0.5, fontSize:15, bold:true, color:WHITE, fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    s.addText(it.name, { x:6.1, y:y+0.55, w:0.5, h:0.4, fontSize:9, color:WHITE, fontFace:"Calibri", align:"center", margin:0 });
    s.addText(`w = ${it.w}`, { x:6.68, y:y+0.03, w:2.85, h:0.3, fontSize:12, bold:true, color:it.color, fontFace:"Calibri", margin:0 });
    s.addText(it.why, { x:6.68, y:y+0.33, w:2.85, h:0.7, fontSize:9.5, color:NAVY, fontFace:"Calibri" });
  });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 10 — REVAC Hyperparameter Tuning
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideHeader(s, "REVAC  —  Principled Hyperparameter Tuning  (Lecture 17)");

  // Relevance table (left)
  card(s, 0.35, 0.88, 4.7, 3.6);
  s.addText("Parameter Relevance  (from final selected pool)", { x:0.5, y:0.93, w:4.3, h:0.35, fontSize:12, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });

  const params = [
    { name:"μ  (pop. size)",        range:"[10,100]", best:"100",    norm:"0.105", rel:"HIGH ★",  rcolor:RED   },
    { name:"λ/μ  (offspring ratio)", range:"[0.5,4]",  best:"4.0",   norm:"0.108", rel:"HIGH ★",  rcolor:RED   },
    { name:"σ_share  (niche radius)",range:"[0.1,0.8]",best:"0.215", norm:"0.123", rel:"HIGH ★",  rcolor:RED   },
    { name:"tournament_k",           range:"[2,7]",    best:"4",     norm:"0.233", rel:"MEDIUM",  rcolor:AMBER },
    { name:"α  (sharing exponent)",  range:"[0.5,2]",  best:"1.64",  norm:"0.434", rel:"low",     rcolor:MUTED },
  ];
  const cols = [0.5, 2.35, 3.25, 3.9, 4.25];
  const hdrs = ["Parameter","Best","Norm. std","Relevance"];
  hdrs.forEach((h, i) => {
    s.addText(h, { x:cols[i+1]-0.05, y:1.35, w:cols[i+1] - (cols[i] || 0) + 0.1, h:0.28, fontSize:9, bold:true, color:MUTED, fontFace:"Calibri", margin:0 });
  });
  s.addShape(pres.shapes.LINE, { x:0.5, y:1.64, w:4.35, h:0, line:{color:"E2E8F0", width:1} });

  params.forEach((p, i) => {
    const y = 1.72 + i * 0.52;
    const rowBg = i % 2 === 0 ? "F8FAFC" : WHITE;
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:y-0.04, w:4.35, h:0.5, fill:{color:rowBg}, line:{color:rowBg} });
    s.addText(p.name,  { x:0.5,  y, w:1.8, h:0.3, fontSize:10, color:NAVY,   fontFace:"Calibri", margin:0 });
    s.addText(p.best,  { x:2.35, y, w:0.9, h:0.3, fontSize:10, bold:true, color:p.rcolor, fontFace:"Calibri", margin:0 });
    s.addText(p.norm,  { x:3.28, y, w:0.7, h:0.3, fontSize:10, color:MUTED, fontFace:"Calibri", margin:0 });
    s.addShape(pres.shapes.RECTANGLE, { x:3.95, y:y+0.01, w:0.75, h:0.24, fill:{color:p.rcolor}, line:{color:p.rcolor} });
    s.addText(p.rel, { x:3.95, y:y+0.01, w:0.75, h:0.24, fontSize:8, bold:true, color:WHITE, fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
  });

  // Best config
  card(s, 0.35, 4.6, 4.7, 0.8, {fill:DARK});
  s.addText("Best configuration  →  python main.py --pop 100 --lambda_ 400 --share 0.2146 --gens 100", {
    x:0.5, y:4.63, w:4.4, h:0.72, fontSize:9, color:TEAL2, fontFace:"Calibri",
  });

  // REVAC convergence image (right)
  const rw = 4.6, rh = rw / 2.25;
  s.addImage({ path:IMG.revac, x:5.2, y:0.88, w:rw, h:rh });

  // Key findings below
  card(s, 5.2, 0.88 + rh + 0.15, 4.6, 3.47 - rh - 0.15 + 1.4);
  const bullets = [
    "μ and λ/μ are HIGH relevance → large diverse populations with many offspring consistently outperform small ones",
    "σ_share HIGH relevance → tight niching (0.215) best; the EA must enforce diversity aggressively",
    "α is irrelevant → sharing mechanism matters; its exact shape does not",
    "Best quality: 2.97 kcal/mol (vs 3.53 default) — a 16% improvement",
  ];
  s.addText("What REVAC Found:", { x:5.35, y:0.88+rh+0.22, w:4.3, h:0.3, fontSize:11, bold:true, color:NAVY, fontFace:"Calibri", margin:0 });
  s.addText(bullets.map((b, i) => ({
    text: b + (i < bullets.length - 1 ? "\n" : ""),
    options: { bullet:true, paraSpaceAfter:4 }
  })), { x:5.35, y:0.88+rh+0.57, w:4.3, h:4.47-rh, fontSize:10, color:NAVY, fontFace:"Calibri" });
}

// ═════════════════════════════════════════════════════════════
// SLIDE 11 — Summary
// ═════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK };

  // Left accent
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:0.09, h:5.625, fill:{color:TEAL}, line:{color:TEAL} });

  s.addText("Key Findings", {
    x:0.35, y:0.12, w:9.3, h:0.65, fontSize:26, bold:true, color:WHITE, fontFace:"Calibri", align:"left", margin:0,
  });

  const findings = [
    {
      icon:"✓", num:"200/200",
      title:"SELFIES Validity Guarantee",
      body:"Every chromosome — including random mutations and crossover products — decodes to a chemically valid molecule. Zero rejections. Solved architecturally, not by post-hoc filtering.",
      color:GREEN,
    },
    {
      icon:"↑", num:"17 vs 3",
      title:"Real Surrogate Resolves Convergence",
      body:"AffinityMLP (RMSE=1.01, R=0.83) produces a meaningful fitness landscape. Archive improvements: 17 with real surrogate vs. ~3 with mock heuristic. Still improving at generation 100.",
      color:TEAL,
    },
    {
      icon:"σ", num:"0.026",
      title:"σᵢ Recovers EGFR Pharmacophore",
      body:"Self-adapted step sizes converged N-heterocycles (imidazole, pyrazine, pyridine, indole) to near-zero σ. The EA independently identified the nitrogen-rich aromatic pharmacophore from binding data alone.",
      color:"0369A1",
    },
    {
      icon:"w", num:"logP #1",
      title:"wⱼ Reveals Constraint Difficulty",
      body:"Evolved penalty weights rank logP (1.72) and PAINS (1.48) as hardest. SA dropped below init (0.74) — SELFIES fragments are pre-synthesizable so SA is rarely violated. EA learned this automatically.",
      color:AMBER,
    },
    {
      icon:"★", num:"HIGH ×3",
      title:"REVAC: μ, λ/μ, σ_share All Critical",
      body:"Three of five parameters are HIGH relevance. α is irrelevant — the sharing mechanism matters, not its shape. Best config (μ=100, λ=400, σ_share=0.215) achieved quality 2.97 kcal/mol.",
      color:"9D174D",
    },
    {
      icon:"≡", num:"4 novel",
      title:"Novel vs. Literature",
      body:"All surveyed GA-based drug design methods use fixed parameters. This project is the first to combine: (1) self-adaptive σᵢ, (2) self-adaptive wⱼ, (3) Tanimoto fitness sharing, and (4) REVAC tuning on a fragment-based EA.",
      color:"7C3AED",
    },
  ];

  findings.forEach((f, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.3 + col * 3.27;
    const y = 0.9 + row * 2.32;

    s.addShape(pres.shapes.RECTANGLE, { x, y, w:3.1, h:2.18, fill:{color:"162B40"}, line:{color:"253F5A"} });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w:3.1, h:0.06, fill:{color:f.color}, line:{color:f.color} });
    s.addText(f.num,   { x:x+0.1, y:y+0.1,  w:3.0, h:0.4, fontSize:17, bold:true, color:f.color, fontFace:"Calibri", align:"left", margin:0 });
    s.addText(f.title, { x:x+0.1, y:y+0.53, w:3.0, h:0.38, fontSize:11, bold:true, color:WHITE,  fontFace:"Calibri", align:"left", margin:0 });
    s.addText(f.body,  { x:x+0.1, y:y+0.93, w:3.0, h:1.15, fontSize:9,  color:"AACFE0", fontFace:"Calibri", align:"left", valign:"top" });
  });
}

// ── Write ─────────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/davisalley/drugea/DrugEA_Presentation.pptx" })
  .then(() => console.log("✓  DrugEA_Presentation.pptx written"))
  .catch(e => { console.error(e); process.exit(1); });
