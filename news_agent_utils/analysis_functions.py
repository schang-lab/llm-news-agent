from typing import Any, Optional
import ast
import numpy as np
import pandas as pd
from scipy import stats
from news_agent_utils.analysis_constants import event_to_frames, frame_alignment_map


def get_frame_score(article_text: str, frame: str, scoring_type: str, client: Optional[Any] = None) -> Optional[int]:
 
    if client is None:
        raise RuntimeError("OpenAI client not provided. Pass the OpenAI client from the notebook as the `client` argument")

    prompt = f"""
How much does this {scoring_type} agree with this perspective?

Perspective:
{frame}

Use the following Likert scale:
-2 = Strongly Disagree
-1 = Disagree
0 = Neutral
1 = Agree
2 = Strongly Agree

Respond with ONLY a single number.

Text:
{article_text}
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        temperature=0,
    )

    output_text = getattr(response, 'output_text', '').strip()

    try:
        return int(output_text)
    except Exception:
        return None


def calculate_avg_frame_scores_for_retrieval_row(row: pd.Series, article_scores_map: Optional[dict] = None) -> pd.Series:
 
    selected_articles_raw = row.get('selected_articles')

    empty = pd.Series({'avg_frame_1_score': np.nan, 'avg_frame_2_score': np.nan,
                       'frame_1_scores': [], 'frame_2_scores': []})

    if pd.isna(selected_articles_raw) or str(selected_articles_raw).strip() == '[]':
        return empty

    try:
        cleaned = str(selected_articles_raw).replace(': nan', ': None').replace(': nan,', ': None,')
        selected_article_items = ast.literal_eval(cleaned)
        if not isinstance(selected_article_items, list):
            raise ValueError("selected_articles is not a list after literal_eval")
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Error parsing selected_articles for retrieval_run_id {row.get('retrieval_run_id')}: {e}")
        return empty

    frame_1_scores = []
    frame_2_scores = []

    for item in selected_article_items:
        current_lookup_id = None
        if isinstance(item, dict) and 'hash_id' in item:
            current_lookup_id = item['hash_id']
        elif isinstance(item, str):
            current_lookup_id = item
        else:
            print(f"Warning: Unexpected type in selected_articles for retrieval_run_id {row.get('retrieval_run_id')}: {type(item)}. Value: {item}")
            continue

        if current_lookup_id:
            scores = (article_scores_map or {}).get(current_lookup_id)
            if scores:
                if pd.notna(scores.get('frame_1_score')):
                    frame_1_scores.append(scores['frame_1_score'])
                if pd.notna(scores.get('frame_2_score')):
                    frame_2_scores.append(scores['frame_2_score'])

    avg_f1 = np.mean(frame_1_scores) if frame_1_scores else np.nan
    avg_f2 = np.mean(frame_2_scores) if frame_2_scores else np.nan

    return pd.Series({
        'avg_frame_1_score': avg_f1,
        'avg_frame_2_score': avg_f2,
        'frame_1_scores':    frame_1_scores,
        'frame_2_scores':    frame_2_scores,
    })


def compute_query_diffs(
    clean_df,
    events_order,
    metadata_type="source+headline",
    frame_cols=("frame_1_score", "frame_2_score"),
    ci_method="bootstrap",
    reorder_frames_by_lean=True,
    n_boot=2000,
    ci=95,
    seed=0,
):
    assert len(frame_cols) == 2, "There should be two frame columns"

    df = clean_df[clean_df["metadata_type"] == metadata_type].copy()

    def _frame_key_for(event, frame_name):
        frames = event_to_frames[event]
        assert frame_name in frames, f"{frame_name!r} not in event_to_frames[{event!r}]"
        if reorder_frames_by_lean:
            return "frame_1" if frame_alignment_map[frame_name] == "liberal" else "frame_2"
        return "frame_1" if frames[0] == frame_name else "frame_2"

    def _target_key(row):
        if row["query_frame"] == "neutral":
            return "neutral"
        return _frame_key_for(row["event"], row["query_frame"])

    df["target_frame"] = df["query_frame"]
    df["target_frame_key"] = df.apply(_target_key, axis=1)
    unmapped = df[df["target_frame_key"].isna()]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} rows had unmapped query_frame; check map")

    biased = df[df["query_type"] != "neutral"]
    biased_means = (
        biased.groupby(["event", "query_type", "target_frame", "target_frame_key", "replicate"])[list(frame_cols)]
        .mean()
        .reset_index()
    )

    neutral = df[df["query_type"] == "neutral"]
    neutral_means = neutral.groupby("event")[list(frame_cols)].mean()

    for fc in frame_cols:
        biased_means[fc] = biased_means.apply(
            lambda r: r[fc] - neutral_means.loc[r["event"], fc], axis=1
        )

    rng = np.random.default_rng(seed)
    lo_p, hi_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    alpha = 1 - ci / 100

    rows = []
    grp_keys = ["event", "query_type", "target_frame", "target_frame_key"]
    for (event, qt, tf, tfk), grp in biased_means.groupby(grp_keys):
        for fi, fc in enumerate(frame_cols):
            scored_frame_name = event_to_frames[event][fi]
            scored_key = _frame_key_for(event, scored_frame_name)

            vals = grp[fc].dropna().to_numpy()
            n = len(vals)
            mean = vals.mean()

            if n < 2:
                lo, hi = mean, mean
            elif ci_method == "bootstrap":
                idx = rng.integers(0, n, size=(n_boot, n))
                lo, hi = np.percentile(vals[idx].mean(axis=1), [lo_p, hi_p])
            elif ci_method == "wald":
                se = vals.std(ddof=1) / np.sqrt(n)
                t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
                lo, hi = mean - t_crit * se, mean + t_crit * se
            else:
                raise ValueError(f"unknown ci_method: {ci_method}")

            rows.append({
                "event": event,
                "query_type": qt,
                "target_frame": tf,
                "target_frame_key": tfk,
                "scored_frame": scored_frame_name,
                "scored_frame_key": scored_key,
                "diff": mean, "ci_lo": lo, "ci_hi": hi,
            })

    out = pd.DataFrame(rows)
    out["event"] = pd.Categorical(out["event"], categories=events_order, ordered=True)
    return out.sort_values([
        "event", "query_type", "target_frame_key", "scored_frame_key"
    ]).reset_index(drop=True)


def compute_neutral_diffs(
    clean_df,
    annotations,
    events_order,
    frame_cols=("frame_1_score", "frame_2_score"),
    reorder_frames_by_lean=True,
    ci_method="bootstrap",
    n_boot=2000,
    ci=95,
    seed=0,
):
    assert len(frame_cols) == 2, "There should be two frame columns"

    neutral = clean_df[clean_df["query_type"] == "neutral"]
    corpus_means = annotations.groupby("event")[list(frame_cols)].mean()

    grp_cols = ["event", "metadata_type", "replicate"]
    retrieved_means = neutral.groupby(grp_cols)[list(frame_cols)].mean().reset_index()

    for fc in frame_cols:
        retrieved_means[fc] = retrieved_means.apply(
            lambda r: r[fc] - corpus_means.loc[r["event"], fc], axis=1
        )

    rng = np.random.default_rng(seed)
    lo_p, hi_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    alpha = 1 - ci / 100

    rows = []
    for (event, mt), grp in retrieved_means.groupby(["event", "metadata_type"]):
        for fi, fc in enumerate(frame_cols):
            frame = event_to_frames[event][fi]
            vals = grp[fc].dropna().to_numpy()
            n = len(vals)
            mean = vals.mean()

            if n < 2:
                lo, hi = mean, mean
            elif ci_method == "bootstrap":
                idx = rng.integers(0, n, size=(n_boot, n))
                boot_means = vals[idx].mean(axis=1)
                lo, hi = np.percentile(boot_means, [lo_p, hi_p])
            elif ci_method == "wald":
                se = vals.std(ddof=1) / np.sqrt(n)
                t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
                lo, hi = mean - t_crit * se, mean + t_crit * se
            else:
                raise ValueError(f"unknown ci_method: {ci_method}")

            if reorder_frames_by_lean:
                if frame_alignment_map[frame] == "liberal":
                    frame_key = "frame_1"
                else:
                    assert frame_alignment_map[frame] == "conservative"
                    frame_key = "frame_2"
            else:
                frame_key = f"frame_{fi+1}"

            rows.append({
                "event": event, "metadata_type": mt, "frame": frame, "frame_key": frame_key,
                "diff": mean, "ci_lo": lo, "ci_hi": hi,
            })

    long = pd.DataFrame(rows)
    long["event"] = pd.Categorical(long["event"], categories=events_order, ordered=True)
    return long.sort_values(["event", "metadata_type", "frame"]).reset_index(drop=True)


def plot_neutral_diffs(diffs, events_order, event_display_names=None, metadata_order=("source+headline", "source", "headline")):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.size"] = 18
    frame1_color = "#0077BB"
    frame2_color = "#CC3399"

    n_events = len(events_order)
    x = np.arange(n_events)
    width = 0.32

    n_rows = len(metadata_order)
    n_rows = 1

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(10, n_events * 1.6), 6.2 * n_rows),
        sharex=True,
        sharey=True,
    )
    if n_rows == 1:
        axes = [axes]

    global_ymin = diffs["ci_lo"].min()
    global_ymax = diffs["ci_hi"].max()
    y_pad = (global_ymax - global_ymin) * 0.20
    global_ymin -= y_pad * 0.3
    global_ymax += y_pad

    for ax, mt in zip(axes, metadata_order):
        if mt == "source+headline":
            ax.axhspan(0, global_ymax, color="#CCEECC", alpha=0.35, zorder=0)
            ax.axhspan(global_ymin, 0, color="#FFCCCC", alpha=0.35, zorder=0)
            ax.grid(True, linestyle="--", alpha=0.6, zorder=1)
            sub = (
                diffs[diffs["metadata_type"] == mt]
                .set_index(["frame_key", "event"])
            )
            f1 = sub.loc["frame_1"].reindex(events_order)
            f2 = sub.loc["frame_2"].reindex(events_order)
            err1 = np.vstack([
                f1["diff"].values - f1["ci_lo"].values,
                f1["ci_hi"].values - f1["diff"].values,
            ])
            err2 = np.vstack([
                f2["diff"].values - f2["ci_lo"].values,
                f2["ci_hi"].values - f2["diff"].values,
            ])
            ax.bar(x - width / 2, f1["diff"].values, width,
                   color=frame1_color,
                   yerr=err1, ecolor="black", capsize=3,
                   error_kw={"linewidth": 0.8, "elinewidth": 1.0},
                   zorder=3,
                   label="Frame 1",
            )
            ax.bar(x + width / 2, f2["diff"].values, width,
                   color=frame2_color,
                   yerr=err2, ecolor="black", capsize=3,
                   error_kw={"linewidth": 0.8, "elinewidth": 1.0},
                   zorder=3,
                   label="Frame 2",
            )
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, zorder=4)
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_title(mt.replace("+", " + ").title(), fontsize=16, fontweight="bold", pad=4)
            ax.set_ylabel("Mean Alignment Score Change\n± 95% CI  (vs. Original Article Set)", fontsize=13)
            ax.tick_params(axis="y", labelsize=11)
            ax.legend(loc="upper right", fontsize=12, title="Frame Score", title_fontsize=12, frameon=True, facecolor="white", framealpha=0.9)

    event_key = (
        diffs[["event", "frame", "frame_key"]]
        .drop_duplicates()
        .set_index("event")
    )

    xtick_labels = []
    for e in events_order:
        display = (event_display_names or {}).get(e, e.replace("_", " ").title())
        rows = event_key.loc[[e]].sort_values("frame_key")
        frames = rows["frame"].values
        if len(frames) == 2:
            f0, f1_ = frames[0], frames[1]
            if f0.endswith("_to_blame"):
                f1l = f0.replace("_to_blame", "")
                f2l = f1_.replace("_to_blame", "")
            else:
                f1l = f0.rsplit("_", 1)[-1]
                f2l = f1_.rsplit("_", 1)[-1]
            label = f"{display}\n({f1l} / {f2l})"
        else:
            label = display
        xtick_labels.append(label)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xtick_labels, rotation=40, ha="right", fontsize=12)
    fig.tight_layout()
    return fig, axes


def draw_heatmap_neutral(ax, mat, title, vmin, vmax, show_xticks=False, 
                          fontsize_annot=13, fontsize_tick=13, row_labels=None, col_labels=None):  
    import numpy as _np
    im = ax.imshow(mat, aspect='auto', cmap='PRGn', vmin=vmin, vmax=vmax)
    n_rows_hm, n_cols_hm = mat.shape
    for x in _np.arange(-0.5, n_cols_hm, 1):
        ax.axvline(x, color='#dddddd', linewidth=0.5)
    for y in _np.arange(-0.5, n_rows_hm, 1):
        ax.axhline(y, color='#dddddd', linewidth=0.5)
    for ri in range(n_rows_hm):
        for ci in range(n_cols_hm):
            val = mat[ri, ci]
            if _np.isnan(val):
                continue
            ax.text(ci, ri, f'{val:.3f}', ha='center', va='center', fontsize=fontsize_annot, color='white' if abs(val) > vmax * 0.6 else 'black')
    
    ax.set_yticks(range(n_rows_hm))
    ax.set_yticklabels(row_labels if row_labels else range(n_rows_hm), fontsize=fontsize_tick)

    if show_xticks:
        ax.set_xticks(range(n_cols_hm))
        ax.set_xticklabels(col_labels if col_labels else range(n_cols_hm), 
                           rotation=45, ha='right', fontsize=fontsize_tick)
    else:
        ax.set_xticks([])

    ax.set_title(title, fontsize=fontsize_tick + 1, pad=8)  

    return im
