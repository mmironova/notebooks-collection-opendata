"""Common loader, selections, weights and plots for the Run 2 vs HL-LHC ttbar comparison.

Both samples are converted to one layout so that every selection is applied identically:

    events.lep        pt, eta, phi, e [GeV], charge, flavour (11/13), medium (ID), iso (Tight_VarRad)
    events.jet        pt, eta, phi, e [GeV], jvt, passes_or, btag (77% WP), hs (1/0, -1 unknown)
    events.truth_lep  pt, eta, phi [GeV], flavour       (dressed, pT > 10 GeV)
    events.truth_jet  pt, eta, phi [GeV]                (AntiKt4TruthDressedWZJets, pT > 20 GeV)
    events.truth_tau_pt                                 (TruthTaus four-vector pT > 10 GeV)
    events.met, met_phi, met_mpx, met_mpy, truth_met, truth_met_mpx, truth_met_mpy [GeV]
    events.mc_weight, events.mu, events.npv             (mu, npv are NaN for Run 2: not stored)

Differences that cannot be removed here (see notebook caveats):
  * Run 2 Open Data keeps only jets with |eta| < 2.5 that passed overlap removal; JVT is a flag.
  * Run 2 b-tagging: DL1dv01, jet_btag_quantile >= 3 (77% WP). HL-LHC: GN2HLv01 has no
    calibration, so the 77% cut is derived from truth-labelled jets in this sample.
  * 410470 contains single-lepton and dilepton ttbar; 601229 is single-lepton only.
"""
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

RUN2_FILE = ('https://opendata.cern.ch/eos/opendata/atlas/rucio/user/egramsta/'
             'mc_410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.noskim.root')
# TODO: not public yet (NERSC path); replace with the Open Data URL once the ntuple is released
HLLHC_FILE = ('/global/cfs/cdirs/atlas/mironova/open_data/OpenData/run/'
              'phase2_matched_10k/output.root')
DATASETS = {
    'run2': 'mc_410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad (Open Data, mc20a)',
    'hllhc': 'mc23_valid.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep e8514_e8586_s4680_r17821',
}
STYLES = {
    'run2': {'label': r'Run 2, 13 TeV', 'color': '#2a78d6', 'marker': 'o', 'linestyle': '-',
             'sample': 'run2'},
    'hllhc': {'label': r'HL-LHC, 13.6 TeV, $\langle\mu\rangle\approx 200$', 'color': '#eb6834',
              'marker': 's', 'linestyle': '--', 'sample': 'hllhc'},
}


SHORT_NAMES = {'run2': 'Run 2', 'hllhc': 'HL-LHC'}
INK, MUTED, GRID = '#262626', '#6f6f6f', '#e6e6e6'


def set_style():
    """Matplotlib defaults for every plot in the notebook: recessive axes and grid, no frames."""
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 150, 'figure.facecolor': 'white',
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.titlelocation': 'left',
        'axes.titlepad': 8, 'axes.labelcolor': INK, 'text.color': INK,
        'axes.edgecolor': '#a0a0a0', 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.8,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.frameon': False, 'legend.fontsize': 10, 'legend.handlelength': 2.2,
        'lines.linewidth': 2, 'lines.markersize': 6,
    })


set_style()


def style(sample, label=None, **changes):
    """Copy of a sample's style with a new label or other changes (keeps the dataset footer)."""
    return {**STYLES[sample], **({'label': label} if label else {}), **changes}
W_MASS = 80.4  # GeV
GN2HL_FC = 0.2  # TODO: c-fraction for the GN2HLv01 discriminant; GN2v01 value used as placeholder
HLLHC_XSEC_PB = None  # TODO: 601229 (e8514) cross section x filter efficiency x k-factor [pb]

RUN2_BRANCHES = [
    'mcWeight', 'xsec', 'kfac', 'filteff',
    'lep_pt', 'lep_eta', 'lep_phi', 'lep_e', 'lep_charge', 'lep_type', 'lep_isMediumID',
    'lep_isTightIso', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_e', 'jet_jvt', 'jet_btag_quantile',
    'truth_elec_pt', 'truth_elec_eta', 'truth_elec_phi', 'truth_muon_pt', 'truth_muon_eta',
    'truth_muon_phi', 'truth_jet_pt', 'truth_jet_eta', 'truth_jet_phi', 'truth_tau_pt',
    'met', 'met_phi', 'met_mpx', 'met_mpy', 'truth_met', 'truth_met_phi',
]


def sort_by_pt(objects):
    return objects[ak.argsort(objects.pt, axis=1, ascending=False)]


def _truth_leptons(electrons, muons):
    leptons = ak.concatenate([electrons, muons], axis=1)
    return sort_by_pt(leptons[leptons.pt > 10])


def load_run2(path=RUN2_FILE, n_events=None):
    a = uproot.open(path, timeout=120)['analysis'].arrays(RUN2_BRANCHES, entry_stop=n_events)
    lep = ak.zip({'pt': a.lep_pt, 'eta': a.lep_eta, 'phi': a.lep_phi, 'e': a.lep_e,
                  'charge': ak.values_astype(a.lep_charge, np.int32),
                  'flavour': ak.values_astype(abs(a.lep_type), np.int32),
                  'medium': a.lep_isMediumID, 'iso': a.lep_isTightIso})
    ones = ak.ones_like(a.jet_pt, dtype=bool)
    jet = ak.zip({'pt': a.jet_pt, 'eta': a.jet_eta, 'phi': a.jet_phi, 'e': a.jet_e,
                  'jvt': a.jet_jvt, 'passes_or': ones,  # Open Data keeps only jets passing OR
                  'btag': a.jet_btag_quantile >= 3,
                  'hs': ak.values_astype(ones, np.int32) * -1})
    truth_e = ak.zip({'pt': a.truth_elec_pt, 'eta': a.truth_elec_eta, 'phi': a.truth_elec_phi,
                      'flavour': ak.full_like(a.truth_elec_pt, 11, dtype=np.int32)})
    truth_m = ak.zip({'pt': a.truth_muon_pt, 'eta': a.truth_muon_eta, 'phi': a.truth_muon_phi,
                      'flavour': ak.full_like(a.truth_muon_pt, 13, dtype=np.int32)})
    nan = np.full(len(a), np.nan, dtype=np.float32)
    events = ak.zip({
        'lep': sort_by_pt(lep), 'jet': sort_by_pt(jet),
        'truth_lep': _truth_leptons(truth_e, truth_m),
        'truth_jet': sort_by_pt(ak.zip({'pt': a.truth_jet_pt, 'eta': a.truth_jet_eta,
                                        'phi': a.truth_jet_phi})),
        'truth_tau_pt': a.truth_tau_pt[a.truth_tau_pt > 10],
        'met': a.met, 'met_phi': a.met_phi, 'met_mpx': a.met_mpx, 'met_mpy': a.met_mpy,
        'truth_met': a.truth_met,
        'truth_met_mpx': a.truth_met * np.cos(a.truth_met_phi),
        'truth_met_mpy': a.truth_met * np.sin(a.truth_met_phi),
        'mc_weight': a.mcWeight, 'mu': nan, 'npv': nan,
    }, depth_limit=1)
    events = ak.with_parameter(events, 'xsec_pb', float(a.xsec[0] * a.kfac[0] * a.filteff[0]))
    return events


def load_hllhc(path=HLLHC_FILE, n_events=None):
    a = uproot.open(path)['analysis'].arrays(entry_stop=n_events)
    gev = 1e-3

    def leptons(prefix, flavour, medium, iso):
        return ak.zip({'pt': a[f'{prefix}_pt_NOSYS'] * gev, 'eta': a[f'{prefix}_eta'],
                       'phi': a[f'{prefix}_phi'], 'e': a[f'{prefix}_e_NOSYS'] * gev,
                       'charge': ak.values_astype(a[f'{prefix}_charge'], np.int32),
                       'flavour': ak.full_like(a[f'{prefix}_eta'], flavour, dtype=np.int32),
                       'medium': a[medium] == 1, 'iso': a[iso] == 1})

    lep = ak.concatenate([
        leptons('elec', 11, 'elec_select_medium_NonIso_NOSYS', 'elec_select_loose_tightiso_NOSYS'),
        # every saved muon passes the Medium baseline
        leptons('muon', 13, 'muon_select_baseline_NOSYS', 'muon_select_medium_tightiso_NOSYS'),
    ], axis=1)
    discriminant = np.log(a.jet_GN2HLv01_pb / (GN2HL_FC * a.jet_GN2HLv01_pc
                                               + (1 - GN2HL_FC) * a.jet_GN2HLv01_pu))
    jet = ak.zip({'pt': a.jet_pt_NOSYS * gev, 'eta': a.jet_eta, 'phi': a.jet_phi,
                  'e': a.jet_e_NOSYS * gev, 'jvt': a.jet_select_baselineJvt_NOSYS == 1,
                  'passes_or': a.jet_select_passesOR_NOSYS == 1, 'btag_score': discriminant,
                  'flavour_label': a.jet_truthflav, 'hs': ak.values_astype(a.jet_isJvtHS, np.int32)})
    jet = sort_by_pt(jet)
    cut = gn2hl_cut(jet)
    jet = ak.zip({'pt': jet.pt, 'eta': jet.eta, 'phi': jet.phi, 'e': jet.e, 'jvt': jet.jvt,
                  'passes_or': jet.passes_or, 'btag': jet.btag_score > cut, 'hs': jet.hs})

    def truth(prefix, flavour):
        return ak.zip({'pt': a[f'truth_{prefix}_pt'] * gev, 'eta': a[f'truth_{prefix}_eta'],
                       'phi': a[f'truth_{prefix}_phi'],
                       'flavour': ak.full_like(a[f'truth_{prefix}_eta'], flavour, dtype=np.int32)})

    truth_jet = sort_by_pt(ak.zip({'pt': a.truth_jet_pt * gev, 'eta': a.truth_jet_eta,
                                   'phi': a.truth_jet_phi}))
    events = ak.zip({
        'lep': sort_by_pt(lep), 'jet': jet,
        'truth_lep': _truth_leptons(truth('elec', 11), truth('muon', 13)),
        'truth_jet': truth_jet[truth_jet.pt > 20],
        'truth_tau_pt': _truth_tau_pt(a),
        'met': a.met_met_NOSYS * gev, 'met_phi': a.met_phi_NOSYS,
        'met_mpx': a.met_mpx * gev, 'met_mpy': a.met_mpy * gev,
        'truth_met': a.truth_met * gev,
        'truth_met_mpx': a.truth_met_mpx * gev, 'truth_met_mpy': a.truth_met_mpy * gev,
        'mc_weight': a.mcWeight, 'mu': a.averageInteractionsPerCrossing,
        'npv': ak.values_astype(ak.sum(a.pv_type != 0, axis=1), np.float32),
    }, depth_limit=1)
    events = ak.with_parameter(events, 'gn2hl_cut', float(cut))
    if HLLHC_XSEC_PB is not None:
        events = ak.with_parameter(events, 'xsec_pb', float(HLLHC_XSEC_PB))
    return events


def _truth_tau_pt(a):
    pt = np.hypot(a.truth_tau_px, a.truth_tau_py) * 1e-3
    return ak.values_astype(pt[pt > 10], np.float32)


def gn2hl_cut(jets, efficiency=0.77):
    """GN2HLv01 discriminant cut giving `efficiency` for truth b-jets in baseline jets.

    Uncalibrated stand-in for a working point: derived from this sample's truth labels."""
    good = (jets.pt > 30) & (abs(jets.eta) < 2.5) & jets.jvt & jets.passes_or
    b_scores = ak.to_numpy(ak.flatten(jets.btag_score[good & (jets.flavour_label == 5)]))
    return np.quantile(b_scores, 1 - efficiency)


def schema(events):
    """Field name -> type string, for every object and event-level field."""
    out = {}
    for field in events.fields:
        array = events[field]
        if array.fields:
            for sub in array.fields:
                out[f'{field}.{sub}'] = str(ak.type(array[sub]).content)
        else:
            out[field] = str(ak.type(array).content)
    return out


def check_schema(run2, hllhc):
    """Assert both samples have identical fields and dtypes after loading."""
    a, b = schema(run2), schema(hllhc)
    missing = sorted(set(a) ^ set(b))
    different = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    assert not missing and not different, f'fields only in one sample: {missing}; dtypes differ: ' + \
        ', '.join(f'{k}: {a[k]} vs {b[k]}' for k in different)
    return a


# ---------------------------------------------------------------- selections (same for both)
def baseline_leptons(events):
    lep = events.lep
    return lep[(lep.pt > 25) & (abs(lep.eta) < 2.5) & lep.medium & lep.iso]


def baseline_jets(events, pt_min=30, eta_max=2.5, jvt=True):
    jet = events.jet
    keep = (jet.pt > pt_min) & (abs(jet.eta) < eta_max) & jet.passes_or
    if jvt:
        keep = keep & jet.jvt
    return jet[keep]


def truth_single_lepton(events):
    """Exactly one dressed truth e/mu (pT > 10 GeV, any eta), inside pT > 25 GeV and
    |eta| < 2.5, and no truth tau with pT > 10 GeV.

    Proxy for 'exactly one leptonic W -> e/mu': the Run 2 ntuple has no truth W record. The
    tau veto removes Run 2 dilepton events with a hadronic tau and, in both samples, W -> tau."""
    tl = events.truth_lep
    one = (ak.num(tl) == 1) & ak.fill_none(ak.firsts((tl.pt > 25) & (abs(tl.eta) < 2.5)), False)
    return one & (ak.num(events.truth_tau_pt) == 0)


def cutflow(events):
    """Ordered (step, mask) pairs of the single-lepton selection."""
    n_lep = ak.num(baseline_leptons(events))
    jets = baseline_jets(events)
    steps = [('all events', ak.ones_like(n_lep, dtype=bool)),
             ('exactly 1 baseline lepton', n_lep == 1)]
    steps.append(('>= 4 baseline jets', steps[-1][1] & (ak.num(jets) >= 4)))
    steps.append(('>= 1 b-tagged jet (77%)', steps[-1][1] & (ak.sum(jets.btag, axis=1) >= 1)))
    return steps


def single_lepton(events):
    return cutflow(events)[2][1]


# ---------------------------------------------------------------- weights
def event_weights(events, lumi_fb):
    """mcWeight x sigma x L / sum(mcWeight over the events read). None if sigma is unknown."""
    xsec = ak.parameters(events).get('xsec_pb')
    if xsec is None:
        return None
    w = ak.to_numpy(events.mc_weight).astype(float)
    return w * xsec * 1e3 * lumi_fb / w.sum()


# ---------------------------------------------------------------- observables
def invariant_mass(objects):
    """Mass of the sum of each event's objects (needs pt, eta, phi, e)."""
    e = ak.sum(objects.e, axis=1)
    px = ak.sum(objects.pt * np.cos(objects.phi), axis=1)
    py = ak.sum(objects.pt * np.sin(objects.phi), axis=1)
    pz = ak.sum(objects.pt * np.sinh(objects.eta), axis=1)
    return np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))


def hadronic_top(jets, method='leading', pt_min_w=0):
    """m_jj of the hadronic W candidate and m_jjb (b-jet maximising pT of the jjb system).

    method='leading': the two highest-pT untagged jets (does not bias the peak).
    method='closest': the untagged pair with m_jj closest to m_W. This piles events up at
    80 GeV by construction, so the peak width no longer measures the jet resolution.
    pt_min_w: minimum pT of the W candidate (None below it).
    None where an event has fewer than two untagged or no b-tagged jets."""
    light, bjets = jets[~jets.btag], jets[jets.btag]
    if method == 'leading':
        light = light[:, :2]
    pairs = ak.combinations(light, 2, fields=['j1', 'j2'])
    m_pair = _pair_mass(pairs.j1, pairs.j2)
    best = ak.argmin(abs(m_pair - W_MASS), axis=1, keepdims=True)
    w = ak.firsts(pairs[best])
    w = ak.mask(w, np.hypot(_sum2(w, np.cos), _sum2(w, np.sin)) > pt_min_w)
    m_jj = _pair_mass(w.j1, w.j2)
    triples = ak.cartesian({'b': bjets, 'w': ak.singletons(w)}, axis=1)
    px = triples.b.pt * np.cos(triples.b.phi) + _sum2(triples.w, np.cos)
    py = triples.b.pt * np.sin(triples.b.phi) + _sum2(triples.w, np.sin)
    pick = ak.argmax(np.hypot(px, py), axis=1, keepdims=True)
    t = ak.firsts(triples[pick])
    m_jjb = _three_mass(t.w.j1, t.w.j2, t.b)
    return m_jj, m_jjb


def _sum2(w, trig):
    return w.j1.pt * trig(w.j1.phi) + w.j2.pt * trig(w.j2.phi)


def _four(j):
    return j.e, j.pt * np.cos(j.phi), j.pt * np.sin(j.phi), j.pt * np.sinh(j.eta)


def _pair_mass(j1, j2):
    e, px, py, pz = (a + b for a, b in zip(_four(j1), _four(j2)))
    return np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))


def _three_mass(j1, j2, j3):
    e, px, py, pz = (a + b + c for a, b, c in zip(_four(j1), _four(j2), _four(j3)))
    return np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))


def fit_peak(values, weights=None, fit_range=(40, 130), n_bins=30,
             means=np.arange(65, 100.01, 0.1), sigmas=np.arange(3, 30.01, 0.1)):
    """Fit Gaussian + linear background to a histogram; return a dict with the result.

    The Gaussian mean and width are scanned on a grid; amplitude and background are solved
    exactly (linear least squares) at each point. Bin variances are the bin contents, so with
    `weights` scaled to a luminosity this is an Asimov fit and the errors (Delta chi2 = 1) are
    the expected statistical precision at that luminosity."""
    edges = np.linspace(*fit_range, n_bins + 1)
    x = (edges[1:] + edges[:-1]) / 2
    y, _ = np.histogram(to_numpy(values), edges, weights=weights)
    var = np.maximum(y, 1.0)
    sw = 1 / np.sqrt(var)
    chi2 = np.empty((len(means), len(sigmas)))
    background = np.broadcast_to(np.column_stack([np.ones_like(x), x - x.mean()]),
                                 (len(sigmas), len(x), 2))
    target = y * sw
    for i, m in enumerate(means):
        gauss = np.exp(-0.5 * ((x[None, :] - m) / sigmas[:, None]) ** 2)  # sigmas x bins
        design = np.concatenate([gauss[..., None], background], axis=2) * sw[None, :, None]
        normal = np.einsum('sbi,sbj->sij', design, design)
        coef = np.linalg.solve(normal, np.einsum('sbi,b->si', design, target)[..., None])
        chi2[i] = np.sum((np.einsum('sbi,si->sb', design, coef[..., 0]) - target) ** 2, axis=1)
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)

    def error(grid, profile, k):
        """Delta chi2 = 1 from a parabola through the profile near its minimum."""
        lo, hi = max(k - 3, 0), min(k + 4, len(grid))
        curvature = np.polyfit(grid[lo:hi] - grid[k], profile[lo:hi] - profile[k], 2)[0]
        return 1 / np.sqrt(curvature) if curvature > 0 else np.nan

    return {'mean': means[i], 'mean_err': error(means, chi2.min(axis=1), i), 'sigma': sigmas[j],
            'sigma_err': error(sigmas, chi2.min(axis=0), j), 'chi2': chi2[i, j], 'ndf': n_bins - 5,
            'at_grid_edge': i in (0, len(means) - 1) or j in (0, len(sigmas) - 1)}


# ---------------------------------------------------------------- plotting
def to_numpy(values):
    """Flatten per-event or per-object values to a 1D float array, dropping missing entries."""
    return ak.to_numpy(ak.flatten(ak.Array(values), axis=None)).astype(float)


def compare(samples, bins, xlabel, title=None, logy=False, density=True, ratio=True,
            show_mean=False, ylabel=None):
    """Overlay [(values, style) or (values, style, weights)] with an optional ratio panel.

    Each sample is a step line with a shaded statistical band (Poisson, sqrt(sum w^2)).
    density=True normalises each histogram to unit area. The ratio panel divides every sample
    by the first; the grey band is the first sample's own uncertainty. Returns the figure."""
    ratio = ratio and len(samples) > 1
    if ratio:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                      gridspec_kw={'height_ratios': [3.2, 1], 'hspace': 0.06})
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.asarray(bins, dtype=float)
    widths = np.diff(edges)
    shapes = []
    for item in samples:
        values, st = to_numpy(item[0]), item[1]
        weights = None if len(item) < 3 or item[2] is None else np.asarray(item[2], dtype=float)
        sumw, _ = np.histogram(values, edges, weights=weights)
        sumw2, _ = np.histogram(values, edges, weights=None if weights is None else weights**2)
        norm = max(sumw.sum(), 1e-12) * widths if density else np.ones_like(widths)
        y, err = sumw / norm, np.sqrt(sumw2) / norm
        shapes.append((y, err))
        label = st['label']
        if show_mean and len(values):
            label += f' (mean {np.average(values, weights=weights):.3g})'
        _band(ax, edges, y - err, y + err, st['color'])
        ax.stairs(y, edges, color=st['color'], linestyle=st['linestyle'], linewidth=2, label=label)
    ax.set_ylabel(ylabel or ('Normalised to unit area' if density else 'Events'))
    if logy:
        ax.set_yscale('log')
    else:
        top = max(np.max(y + e) for y, e in shapes)
        ax.set_ylim(0, 1.3 * top)  # headroom for the legend
    ax.legend(loc='upper right')
    if title:
        ax.set_title(title)
    ax.set_xlim(edges[0], edges[-1])
    if ratio:
        ref, ref_err = shapes[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(ref > 0, ref_err / ref, np.nan)
            _band(rax, edges, 1 - rel, 1 + rel, MUTED, alpha=0.25)
            ratios = []
            for (y, err), item in zip(shapes[1:], samples[1:]):
                r = np.where(ref > 0, y / ref, np.nan)
                _band(rax, edges, r - err / ref, r + err / ref, item[1]['color'])
                rax.stairs(r, edges, color=item[1]['color'], linestyle=item[1]['linestyle'],
                           linewidth=2)
                ratios.append(r[np.isfinite(r) & (rel < 0.3)])
        rax.axhline(1, color=MUTED, linewidth=1)
        finite = np.concatenate(ratios) if ratios else np.array([1.0])
        lo = min(0.5, np.percentile(finite, 5) - 0.1) if finite.size else 0.5
        hi = max(1.5, np.percentile(finite, 95) + 0.1) if finite.size else 1.5
        rax.set_ylim(max(lo, 0), min(hi, 3))
        reference = SHORT_NAMES.get(samples[0][1].get('sample'), samples[0][1]['label'])
        rax.set_ylabel(f'Ratio to {reference}', fontsize=9)
        rax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    footer(fig, [item[1].get('sample') for item in samples])
    return fig


def _band(ax, edges, low, high, color, alpha=0.18):
    """Shaded band following a step histogram."""
    ax.fill_between(edges, np.append(low, low[-1]), np.append(high, high[-1]), step='post',
                    color=color, alpha=alpha, linewidth=0)


def footer(fig, samples):
    """Dataset names of the samples shown, small and muted, under the figure."""
    keys = [k for k in DATASETS if k in samples]
    if keys:
        fig.text(0.01, 0.005, '\n'.join(DATASETS[k] for k in keys), fontsize=6.5, color=MUTED,
                 va='bottom')
        fig.subplots_adjust(bottom=0.11 + 0.022 * len(keys))


def shade(ax, lo, hi, text=None, color='#f1f1f1', y=0.03):
    """Light background band from lo to hi on the x axis, with an optional small label."""
    ax.axvspan(lo, hi, color=color, zorder=0, linewidth=0)
    if text:
        ax.text((lo + hi) / 2, y, text, transform=ax.get_xaxis_transform(), ha='center',
                va='bottom', fontsize=8.5, color=MUTED)


def vline(ax, x, text=None, color=MUTED, y=0.97, ha='left', linestyle='--'):
    """Vertical reference line with a small label next to its top."""
    ax.axvline(x, color=color, linewidth=1.2, linestyle=linestyle, zorder=1)
    if text:
        pad = ' ' if ha == 'left' else ''
        ax.text(x, y, f'{pad}{text}{" " if ha == "right" else ""}', transform=ax.get_xaxis_transform(),
                ha=ha, va='top', fontsize=9, color=INK)
