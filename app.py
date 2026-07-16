import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import date

st.set_page_config(page_title="Dynamic Gantt Chart", layout="wide")
st.title("📊 Dynamic Gantt Chart Generator")

REQUIRED_COLUMNS = ["Task ID", "Task Name", "Type", "Resource", "Priority", "Parent Task ID",
                    "Start Date", "End Date", "Duration (Days)", "% Complete", "Dependencies"]

COLOR_SCHEMES = {
    "Vibrant (default)": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "Tableau 10": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
    "Corporate Blue": ["#1F4E79", "#2E75B6", "#9DC3E6", "#548235", "#BF8F00", "#C00000", "#7030A0", "#525252"],
    "Modern (Validated)": ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "Pastel": ["#A8D8EA", "#AA96DA", "#FCBAD3", "#FFE3A3", "#B5EAD7", "#C7CEEA", "#FFDAC1", "#E2F0CB"],
    "Monochrome Navy": ["#0B3D91", "#1F5FA8", "#3C7DC0", "#6FA3D8", "#A9C9EA", "#14294A", "#4B6C9E", "#7E97B8"],
}


# ----------------------------- Core scheduling logic (no st.* calls, unit-testable) -----------------------------
def _busday(d):
    return np.datetime64(pd.Timestamp(d).date())


def validate_tasks(df):
    errors = []
    ids = df['Task ID'].astype(str).str.strip()
    blanks = df[ids == ""]
    if not blanks.empty:
        errors.append(f"{len(blanks)} row(s) have a blank Task ID. Every task needs a unique ID.")
    non_blank_ids = ids[ids != ""]
    dupes = non_blank_ids[non_blank_ids.duplicated()].unique().tolist()
    if dupes:
        errors.append(f"Duplicate Task ID(s) found: {', '.join(dupes)}. Task IDs must be unique.")
    return errors


def resolve_schedule(df, project_start_date):
    df = df.copy()
    df['Dependencies'] = df['Dependencies'].fillna("").astype(str)
    df['Parent Task ID'] = df.get('Parent Task ID', "").fillna("").astype(str).str.strip()
    df['% Complete'] = pd.to_numeric(df.get('% Complete', 0), errors='coerce').fillna(0).clip(0, 100)

    parent_ids = set(df.loc[df['Parent Task ID'] != "", 'Parent Task ID'])
    all_ids = set(df['Task ID'].astype(str).str.strip())
    parent_ids &= all_ids

    leaf_df = df[~df['Task ID'].astype(str).str.strip().isin(parent_ids)]

    task_dict, leaf_order = {}, []
    pending_tasks = leaf_df.to_dict('records')
    valid_project_start = pd.to_datetime(np.busday_offset(_busday(project_start_date), 0, roll='forward')).date()

    progress_made = True
    while pending_tasks and progress_made:
        progress_made = False
        remaining = []
        for task in pending_tasks:
            task_id = str(task['Task ID']).strip()
            deps = [d.strip() for d in task['Dependencies'].split(',') if d.strip()]

            if deps and not all(d in task_dict for d in deps):
                remaining.append(task)
                continue

            manual_start, manual_end, manual_dur = task.get('Start Date'), task.get('End Date'), task.get('Duration (Days)')

            if not deps:
                start_date = pd.to_datetime(manual_start).date() if pd.notna(manual_start) and manual_start != "" else valid_project_start
            else:
                start_date = max(task_dict[d]['end_date'] for d in deps)

            start_date = pd.to_datetime(np.busday_offset(_busday(start_date), 0, roll='forward')).date()
            start_np = _busday(start_date)

            if pd.notna(manual_end) and manual_end != "":
                end_date = pd.to_datetime(manual_end).date()
                duration = max(0, int(np.busday_count(start_np, _busday(end_date))))
            else:
                duration = int(manual_dur) if pd.notna(manual_dur) else 1
                end_date = pd.to_datetime(np.busday_offset(start_np, duration)).date()

            task_dict[task_id] = {
                'start_date': start_date, 'end_date': end_date, 'duration': duration,
                'pct': float(task.get('% Complete', 0) or 0), 'is_summary': False, 'deps': deps,
            }
            leaf_order.append(task_id)
            progress_made = True
        pending_tasks = remaining

    remaining_parents = list(parent_ids)
    safety = 0
    while remaining_parents and safety < 15:
        safety += 1
        still_remaining = []
        for pid in remaining_parents:
            child_ids = df.loc[df['Parent Task ID'] == pid, 'Task ID'].astype(str).str.strip().tolist()
            child_recs = [task_dict[c] for c in child_ids if c in task_dict]
            if child_ids and len(child_recs) == len(child_ids):
                starts, ends = [c['start_date'] for c in child_recs], [c['end_date'] for c in child_recs]
                durations, pcts = [c['duration'] for c in child_recs], [c['pct'] for c in child_recs]
                total_dur = sum(durations) or 1
                weighted_pct = sum(p * d for p, d in zip(pcts, durations)) / total_dur
                task_dict[pid] = {
                    'start_date': min(starts), 'end_date': max(ends),
                    'duration': int(np.busday_count(_busday(min(starts)), _busday(max(ends)))),
                    'pct': round(weighted_pct, 1), 'is_summary': True, 'deps': [],
                }
            else:
                still_remaining.append(pid)
        if len(still_remaining) == len(remaining_parents):
            break
        remaining_parents = still_remaining

    unresolved = (all_ids - parent_ids) - set(leaf_order)
    return task_dict, leaf_order, unresolved, parent_ids


def compute_critical_path(task_dict, leaf_order):
    if not leaf_order:
        return set()
    successors = {tid: [] for tid in leaf_order}
    for tid in leaf_order:
        for d in task_dict[tid]['deps']:
            if d in successors:
                successors[d].append(tid)

    project_end = max(task_dict[t]['end_date'] for t in leaf_order)
    LF, LS = {}, {}
    for tid in reversed(leaf_order):
        succs = successors.get(tid, [])
        LF[tid] = min((LS[s] for s in succs), default=project_end) if succs else project_end
        duration = task_dict[tid]['duration']
        LS[tid] = pd.to_datetime(np.busday_offset(_busday(LF[tid]), -duration, roll='backward')).date() if duration > 0 else LF[tid]

    return {tid for tid in leaf_order if np.busday_count(_busday(task_dict[tid]['start_date']), _busday(LS[tid])) <= 0}


def build_wbs_order(df):
    """Depth-first order (parents immediately followed by children) + indent depth per Task ID."""
    df = df.copy()
    df['Parent Task ID'] = df.get('Parent Task ID', "").fillna("").astype(str).str.strip()
    children_of = {}
    for _, row in df.iterrows():
        children_of.setdefault(row['Parent Task ID'], []).append(str(row['Task ID']).strip())

    order, depth = [], {}

    def visit(pid, d, seen):
        for cid in children_of.get(pid, []):
            if cid in seen:
                continue
            seen.add(cid)
            order.append(cid)
            depth[cid] = d
            visit(cid, d + 1, seen)

    visit("", 0, set())
    all_ids = df['Task ID'].astype(str).str.strip().tolist()
    for tid in all_ids:
        if tid not in depth:
            order.append(tid)
            depth[tid] = 0
    return order, depth


def wrap_label(text, width=26):
    """Break a long label into multiple lines (<br>) so large fonts don't run off the chart edge."""
    words, lines, cur = str(text).split(' '), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


# --------------------------------------------------------- Sidebar ---------------------------------------------------------
with st.sidebar:
    st.header("📂 Manage Projects")
    st.session_state.setdefault('data_version', 0)

    uploaded_file = st.file_uploader("Upload a saved .csv file", type=["csv"])
    if uploaded_file is not None and uploaded_file.name != st.session_state.get('last_uploaded'):
        try:
            df_up = pd.read_csv(uploaded_file)
            for col in REQUIRED_COLUMNS:
                if col not in df_up.columns:
                    df_up[col] = "" if col not in ("Duration (Days)", "% Complete") else 0
            df_up['Start Date'] = pd.to_datetime(df_up['Start Date'], errors='coerce')
            df_up['End Date'] = pd.to_datetime(df_up['End Date'], errors='coerce')
            st.session_state.task_data = df_up
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.data_version += 1  # forces data_editor to remount with fresh data
            st.success("Project loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")

    st.divider()
    st.subheader("Templates")
    template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    st.download_button("📄 Download CSV Template", template_df.to_csv(index=False).encode('utf-8'),
                        "gantt_template.csv", "text/csv", use_container_width=True, key="template_dl_btn")

    st.divider()
    st.header("🗂️ Grouping & Hierarchy")
    group_mode = st.radio("Layout mode", ["Flat list", "Swimlanes by Resource", "WBS Hierarchy (Parent/Child)"], label_visibility="collapsed")

    st.divider()
    st.header("↕️ Task Order")
    task_order_choice = st.radio(
        "Vertical order of rows", ["First row at top", "First row at bottom"],
        index=0, label_visibility="collapsed",
    )

    st.divider()
    st.header("🎨 Conditional Formatting")

    st.divider()
    st.header("✅ Progress & Critical Path")
    show_pct_overlay = st.checkbox("Show % Complete overlay", value=True)
    show_critical_path = st.checkbox("Highlight critical path", value=True)
    show_today = st.checkbox("Show 'Today' marker line", value=True)

    st.divider()
    st.header("🖇️ Connector Styling")
    conn_color = st.color_picker("Connector Color", "#888888")
    conn_width = st.slider("Connector Thickness", 1, 5, 2)
    conn_style = st.selectbox("Connector Style", ["dot", "dash", "solid"])

    st.divider()
    st.header("🔤 Font Sizes")
    axis_title_size = st.slider("Axis Title Font Size", 10, 28, 16)
    axis_tick_size = st.slider("Axis Tick Label Font Size (dates)", 8, 24, 12)
    task_label_size = st.slider("Task Label Font Size (Y-axis)", 8, 24, 13)
    wrap_labels = st.checkbox("Wrap long task labels", value=True,
                              help="Breaks task names onto two lines instead of one long line, so big fonts don't get cut off.")

    st.divider()
    st.header("📐 Chart Size & Guides")
    fit_width = st.checkbox("Fit chart to window width", value=True)
    chart_width = st.slider("Chart width (px)", 600, 2400, 1100, 50, disabled=fit_width,
                            help="Uncheck 'Fit chart to window width' to set an exact width. Narrower charts are easier to scan when you have many tasks.")
    row_height = st.slider("Row height (px per task)", 24, 90, 45, 2,
                           help="Vertical space each task gets. Lower = more compact chart.")
    show_zebra = st.checkbox("Zebra row shading", value=True,
                             help="Shades alternate rows so the eye can follow a label across to its bar.")
    show_row_lines = st.checkbox("Dotted row guide lines", value=False,
                                 help="A faint dotted line across every row. Off by default — with wrapped labels it can look busy.")

# --------------------------------------------------------- Data init ---------------------------------------------------------
if 'task_data' not in st.session_state:
    st.session_state.task_data = pd.DataFrame({
        "Task ID": ["T1", "T2", "P1", "C1", "C2", "M1", "T3", "T4", "G1"],
        "Task Name": ["Project Scoping", "Kickoff Meeting", "Design Phase", "Wireframes", "Visual Design",
                      "Design Approval", "Backend Dev", "Frontend Dev", "Launch Decision"],
        "Type": ["Task", "Task", "Task", "Task", "Task", "Milestone", "Task", "Task", "Go/No-Go"],
        "Resource": ["Alice", "Alice", "Bob", "Bob", "Bob", "Client", "Charlie", "Alice", "Stakeholders"],
        "Priority": ["High", "Medium", "High", "Medium", "Medium", "High", "High", "Medium", "Critical"],
        "Parent Task ID": ["", "", "", "P1", "P1", "", "", "", ""],
        "Start Date": [date.today(), pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "End Date": [pd.NaT] * 9,
        "Duration (Days)": [3, 1, 0, 5, 6, 0, 7, 6, 0],
        "% Complete": [100, 100, 0, 40, 10, 0, 0, 0, 0],
        "Dependencies": ["", "T1", "", "T2", "C1", "P1", "M1", "M1", "T3, T4"],
    })
    # Coerce dates ONCE at init. Re-coercing on every rerun mutates the editor's input
    # DataFrame, which resets st.data_editor's in-progress state and eats the first edit.
    st.session_state.task_data['Start Date'] = pd.to_datetime(st.session_state.task_data['Start Date'], errors='coerce')
    st.session_state.task_data['End Date'] = pd.to_datetime(st.session_state.task_data['End Date'], errors='coerce')
    for col in REQUIRED_COLUMNS:
        if col not in st.session_state.task_data.columns:
            st.session_state.task_data[col] = "" if col not in ("Duration (Days)", "% Complete") else 0

# --------------------------------------------------------- Editable grid ---------------------------------------------------------
st.subheader("1. Edit Your Tasks")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("Modify dates, durations, or dependencies. **Enter End Date OR Duration.** Set **Parent Task ID** for WBS sub-tasks.")
with col2:
    project_start = st.date_input("Global Project Start", date.today())
with col3:
    timeline_view = st.selectbox("Timeline View", ["Auto", "Weeks", "Months", "Quarters"])
    theme_choice = st.radio("Chart Theme", ["Light", "Dark"], horizontal=True)
    year_band = st.checkbox("Year row under axis", value=True, disabled=timeline_view not in ("Months", "Quarters"),
                            help="Months/Quarters view only: axis shows just 'Jan, Feb…' and each year is written once "
                                 "below, with a line spanning its months.")

edited_df = st.data_editor(
    st.session_state.task_data,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key=f"editor_state_{st.session_state.data_version}",
    column_order=REQUIRED_COLUMNS,
    column_config={
        "Task ID": st.column_config.TextColumn(required=True),
        "Task Name": st.column_config.TextColumn(required=True),
        "Type": st.column_config.SelectboxColumn("Type", options=["Task", "Milestone", "Go/No-Go"], required=True),
        "Resource": st.column_config.TextColumn(),
        "Priority": st.column_config.SelectboxColumn("Priority", options=["Low", "Medium", "High", "Critical"]),
        "Parent Task ID": st.column_config.TextColumn("Parent ID", help="Leave blank for a top-level task. Set to another Task ID to nest it as a sub-task (WBS)."),
        "Start Date": st.column_config.DateColumn("Start Date"),
        "End Date": st.column_config.DateColumn("End Date"),
        "Duration (Days)": st.column_config.NumberColumn("Duration", min_value=0),
        "% Complete": st.column_config.NumberColumn("% Complete", min_value=0, max_value=100),
        "Dependencies": st.column_config.TextColumn(),
    }
)
# NOTE: do not write edited_df back into st.session_state.task_data here — the editor
# persists its own edits via its key, and feeding the mutated frame back in as input
# forces the widget to reset, so users had to type every change twice.


def _same(a, b):
    return (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and a == b)


def sync_dates_durations(df, prev_by_id):
    """Two-way fill between End Date and Duration for rows with a Start Date.

    Uses the previous run's values to tell WHICH field the user just edited, so an
    edited Duration recomputes End Date and an edited End Date recomputes Duration —
    never fighting the user's latest entry."""
    df = df.copy()
    changed = False
    for idx in df.index:
        start, end, dur = df.at[idx, 'Start Date'], df.at[idx, 'End Date'], df.at[idx, 'Duration (Days)']
        if pd.isna(start):
            continue  # dependency-driven rows: the scheduler owns their dates
        prow = prev_by_id.get(str(df.at[idx, 'Task ID']).strip())
        end_edited = prow is not None and not _same(end, prow['End Date'])
        dur_edited = prow is not None and not _same(dur, prow['Duration (Days)'])
        try:
            if pd.notna(dur) and dur_edited and not end_edited:
                new_end = pd.Timestamp(np.busday_offset(_busday(start), int(dur), roll='forward'))
                if not _same(end, new_end):
                    df.at[idx, 'End Date'] = new_end
                    changed = True
            elif pd.notna(end):
                d = max(0, int(np.busday_count(_busday(start), _busday(end))))
                if pd.isna(dur) or int(dur) != d:
                    df.at[idx, 'Duration (Days)'] = d
                    changed = True
            elif pd.notna(dur):  # Start + Duration, no End yet → fill End
                df.at[idx, 'End Date'] = pd.Timestamp(np.busday_offset(_busday(start), int(dur), roll='forward'))
                changed = True
        except Exception:
            continue
    return df, changed


_prev = st.session_state.get('grid_snapshot')
_prev_by_id = ({str(r['Task ID']).strip(): r for _, r in _prev.iterrows()} if _prev is not None else {})
synced_df, needs_sync = sync_dates_durations(edited_df, _prev_by_id)
st.session_state.grid_snapshot = synced_df.copy()
if needs_sync:
    # Push the computed cells into the table and remount the editor so the user sees them
    # immediately. The second pass computes identical values, so this can't rerun-loop.
    st.session_state.task_data = synced_df
    st.session_state.data_version += 1
    st.rerun()
edited_df = synced_df

with st.sidebar:
    st.divider()
    csv_data = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Active Project", csv_data, "my_gantt_project.csv", "text/csv",
                        use_container_width=True, key="active_dl_btn")

# --------------------------------------------------------- Validate ---------------------------------------------------------
clean_df = edited_df.dropna(subset=['Task ID']).copy()
clean_df = clean_df[clean_df['Task ID'].astype(str).str.strip() != ""]
validation_errors = validate_tasks(edited_df.dropna(subset=['Task ID']))

st.subheader("2. Project Timeline")

if validation_errors:
    for err in validation_errors:
        st.error(f"⚠️ {err}")
    st.stop()

if clean_df.empty:
    st.info("Add tasks and valid durations to generate the timeline.")
    st.stop()

# --------------------------------------------------------- Scheduling (guarded) ---------------------------------------------------------
try:
    task_dict, leaf_order, unresolved, parent_ids = resolve_schedule(clean_df, project_start)
except Exception as e:
    st.error(f"⚠️ Couldn't compute the schedule — check dates, durations, and dependency IDs for typos. Details: {e}")
    st.stop()

if unresolved:
    st.error(f"⚠️ Some tasks could not be scheduled (check for circular or missing dependency IDs): {', '.join(sorted(unresolved))}")

clean_df['Start Date'] = clean_df['Task ID'].astype(str).str.strip().map(lambda x: task_dict.get(x, {}).get('start_date'))
clean_df['End Date'] = clean_df['Task ID'].astype(str).str.strip().map(lambda x: task_dict.get(x, {}).get('end_date'))
clean_df['Duration (Days)'] = clean_df['Task ID'].astype(str).str.strip().map(lambda x: task_dict.get(x, {}).get('duration', 1))
clean_df['% Complete'] = clean_df['Task ID'].astype(str).str.strip().map(lambda x: task_dict.get(x, {}).get('pct', 0))
clean_df['Is Summary'] = clean_df['Task ID'].astype(str).str.strip().isin(parent_ids)

valid_df = clean_df.dropna(subset=['Start Date', 'End Date']).copy()

if valid_df.empty:
    st.info("Add tasks and valid durations to generate the timeline.")
    st.stop()

try:
    critical_ids = compute_critical_path(task_dict, leaf_order) if show_critical_path else set()
    if show_critical_path and critical_ids:
        child_map = valid_df.groupby(valid_df['Parent Task ID'])['Task ID'].apply(list).to_dict()
        for pid in parent_ids:
            if any(c in critical_ids for c in child_map.get(pid, [])):
                critical_ids.add(pid)
except Exception as e:
    st.warning(f"Critical path could not be computed: {e}")
    critical_ids = set()

# Project summary strip
leaf_rows = valid_df[~valid_df['Is Summary']]
_durs = leaf_rows['Duration (Days)'].fillna(0).astype(float)
overall_pct = (leaf_rows['% Complete'].fillna(0) * _durs).sum() / max(_durs.sum(), 1)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Project Start", pd.Timestamp(valid_df['Start Date'].min()).strftime("%d %b %Y"))
m2.metric("Project End", pd.Timestamp(valid_df['End Date'].max()).strftime("%d %b %Y"))
m3.metric("Tasks", f"{len(leaf_rows)}")
m4.metric("Overall Complete", f"{overall_pct:.0f}%")

# --------------------------------------------------------- Layout ordering (Flat / Swimlane / WBS) ---------------------------------------------------------
if group_mode == "WBS Hierarchy (Parent/Child)":
    order, depth = build_wbs_order(valid_df)
    id_to_row = {str(r['Task ID']).strip(): r for _, r in valid_df.iterrows()}
    order = [tid for tid in order if tid in id_to_row]
    valid_df = valid_df.set_index(valid_df['Task ID'].astype(str).str.strip()).loc[order].reset_index(drop=True)
    valid_df['_display'] = valid_df.apply(lambda r: ("　" * 2 * depth.get(str(r['Task ID']).strip(), 0)) +
                                           ("🗂️ " if r['Is Summary'] else "") + str(r['Task Name']), axis=1)
elif group_mode == "Swimlanes by Resource":
    valid_df = valid_df.sort_values(by=['Resource', 'Start Date'], kind='stable').reset_index(drop=True)
    valid_df['_display'] = valid_df['Task Name']
else:
    valid_df['_display'] = valid_df['Task Name']

if wrap_labels:
    valid_df['_display'] = valid_df['_display'].map(wrap_label)

display_order = valid_df['_display'].tolist()

# --------------------------------------------------------- Conditional formatting (color-by) ---------------------------------------------------------
categorical_cols = [c for c in ["Resource", "Priority", "Type"] if c in valid_df.columns]
with st.sidebar:
    color_by = st.selectbox("Color bars by", categorical_cols, index=0, key="color_by_select")
    categories = sorted(valid_df[color_by].dropna().astype(str).unique().tolist())
    scheme_name = st.selectbox("Color Scheme", list(COLOR_SCHEMES.keys()), index=0, key="color_scheme_select")
    palette = COLOR_SCHEMES[scheme_name]
    color_map = {}
    with st.expander(f"Customize '{color_by}' colors"):
        for i, cat in enumerate(categories):
            default_color = palette[i % len(palette)]
            color_map[cat] = st.color_picker(cat, default_color, key=f"color_{color_by}_{cat}_{scheme_name}")

# --------------------------------------------------------- Build figure ---------------------------------------------------------
plotly_template = "plotly_dark" if theme_choice == "Dark" else "plotly_white"

fig = px.timeline(
    valid_df, x_start="Start Date", x_end="End Date", y="_display", color=color_by,
    color_discrete_map=color_map,
    hover_data=["Task ID", "Type", "Resource", "Priority", "Duration (Days)", "% Complete", "Dependencies"],
    category_orders={"_display": display_order},
    template=plotly_template,
)
fig.update_yaxes(autorange="reversed" if task_order_choice == "First row at top" else True)
fig.update_layout(barmode="overlay")

# % Complete overlay (a darker inner fill showing progress within each bar)
if show_pct_overlay:
    for _, row in valid_df.iterrows():
        pct = max(0, min(100, row.get('% Complete', 0) or 0))
        if pct <= 0 or row['Type'] == 'Milestone':
            continue
        start, end = row['Start Date'], row['End Date']
        complete_ms = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() * 1000 * (pct / 100.0)
        fig.add_trace(go.Bar(
            x=[complete_ms], y=[row['_display']], base=[start],
            orientation='h', width=0.35,
            marker=dict(color='rgba(0,0,0,0.4)'),
            showlegend=False, hoverinfo='skip',
        ))

# Critical path outline
if show_critical_path:
    for _, row in valid_df.iterrows():
        if str(row['Task ID']).strip() in critical_ids:
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=row['Start Date'], x1=row['End Date'],
                y0=row['_display'], y1=row['_display'],
                line=dict(color="crimson", width=3), opacity=0.9,
            )

# Dependency connector lines
for _, row in valid_df.iterrows():
    deps_str = str(row.get('Dependencies', '') or '')
    if deps_str.strip():
        for d in [x.strip() for x in deps_str.split(',') if x.strip()]:
            dep_rows = valid_df[valid_df['Task ID'].astype(str).str.strip() == d]
            if not dep_rows.empty:
                dep_row = dep_rows.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[dep_row['End Date'], row['Start Date']], y=[dep_row['_display'], row['_display']],
                    mode='lines+markers', line=dict(shape='hv', color=conn_color, width=conn_width, dash=conn_style),
                    marker=dict(symbol='circle', size=[0, 6], color=conn_color),
                    showlegend=False, hoverinfo='skip',
                ))
    if row.get('Type') == 'Milestone':
        fig.add_trace(go.Scatter(x=[row['End Date']], y=[row['_display']], mode='markers',
                                  marker=dict(symbol='star', size=20, color='gold', line=dict(width=1, color='black')),
                                  showlegend=False, hoverinfo='skip'))
    elif row.get('Type') == 'Go/No-Go':
        fig.add_trace(go.Scatter(x=[row['End Date']], y=[row['_display']], mode='markers',
                                  marker=dict(symbol='diamond', size=18, color='crimson', line=dict(width=1, color='black')),
                                  showlegend=False, hoverinfo='skip'))

# Swimlane bands
if group_mode == "Swimlanes by Resource":
    band_colors = ["rgba(128,128,128,0.08)", "rgba(128,128,128,0.16)"]
    pos = 0
    for i, (resource, group) in enumerate(valid_df.groupby('Resource', sort=False)):
        n = len(group)
        fig.add_hrect(y0=pos - 0.5, y1=pos + n - 0.5, fillcolor=band_colors[i % 2], line_width=0, layer="below")
        fig.add_annotation(x=0, xref="paper", y=pos + n / 2 - 0.5, yref="y", text=f"<b>{resource}</b>",
                            showarrow=False, xanchor="right", xshift=-10, font=dict(size=task_label_size))
        pos += n

# Row guides: zebra stripes and/or a faint dotted line at each row tie the label to its bar
if show_zebra and group_mode != "Swimlanes by Resource":  # swimlanes already have their own banding
    for i in range(1, len(valid_df), 2):
        fig.add_hrect(y0=i - 0.5, y1=i + 0.5, fillcolor="rgba(128,128,128,0.08)",
                      line_width=0, layer="below")
if show_row_lines:
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.35)", griddash="dot")
else:
    fig.update_yaxes(showgrid=False)

use_year_band = year_band and timeline_view in ("Months", "Quarters")
if timeline_view == "Months":
    fig.update_xaxes(dtick="M1", tickformat="%b" if use_year_band else "%b %Y")
elif timeline_view == "Quarters":
    fig.update_xaxes(dtick="M3", tickformat="Q%q" if use_year_band else "Q%q %Y")
elif timeline_view == "Weeks":
    fig.update_xaxes(dtick=604800000, tickformat="%W %Y")

bg_color = "#111111" if theme_choice == "Dark" else "#FFFFFF"
text_color = "#FFFFFF" if theme_choice == "Dark" else "#000000"

# Rows must be at least tall enough for the label font (and its wrapped lines), or
# large fonts make adjacent labels overlap regardless of the row-height slider.
max_label_lines = max((str(t).count("<br>") + 1 for t in display_order), default=1)
row_px = max(row_height, int(task_label_size * 1.5 * max_label_lines) + 12)
top_margin = 40
bottom_margin = 60 + int(axis_tick_size * 2.4) if use_year_band else 20
chart_height = max(400, len(valid_df) * row_px)

fig.update_layout(
    xaxis_title="Timeline", yaxis_title="", height=chart_height,
    margin=dict(l=140 if group_mode == "Swimlanes by Resource" else 20, r=20, t=top_margin, b=bottom_margin),
    showlegend=True, paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color),
)
if not fit_width:
    fig.update_layout(width=chart_width)
# automargin grows the margins to fit whatever font size the user picks, so labels never clip
fig.update_xaxes(tickfont=dict(size=axis_tick_size), title_font=dict(size=axis_title_size), automargin=True)
fig.update_yaxes(tickfont=dict(size=task_label_size), title_font=dict(size=axis_title_size), automargin=True)

date_min = pd.Timestamp(valid_df['Start Date'].min())
date_max = pd.Timestamp(valid_df['End Date'].max())

# Year band: month ticks show only "Jan, Feb…" and each year is written ONCE below,
# under a line spanning that year's months.
if use_year_band:
    plot_h = max(1, chart_height - top_margin - bottom_margin)
    y_line = -(axis_tick_size + 16) / plot_h
    y_text = y_line - (axis_tick_size + 8) / plot_h
    band_color = "rgba(160,160,160,0.9)"
    for yr in range(date_min.year, date_max.year + 1):
        seg0 = max(date_min, pd.Timestamp(yr, 1, 1))
        seg1 = min(date_max, pd.Timestamp(yr, 12, 31))
        if seg0 >= seg1:
            continue
        fig.add_shape(type="line", xref="x", yref="paper", x0=seg0, x1=seg1, y0=y_line, y1=y_line,
                      line=dict(color=band_color, width=2))
        fig.add_annotation(x=seg0 + (seg1 - seg0) / 2, xref="x", y=y_text, yref="paper", yanchor="top",
                           text=f"<b>{yr}</b>", showarrow=False,
                           font=dict(size=axis_tick_size, color=text_color))

# 'Today' marker (only drawn when today falls inside the plotted range)
if show_today and date_min <= pd.Timestamp(date.today()) <= date_max:
    today_ts = pd.Timestamp(date.today())
    fig.add_shape(type="line", xref="x", yref="paper", x0=today_ts, x1=today_ts, y0=0, y1=1,
                  line=dict(color="#FF6B6B", width=2, dash="dash"))
    fig.add_annotation(x=today_ts, xref="x", y=1, yref="paper", yanchor="bottom", text="Today",
                       showarrow=False, font=dict(size=max(10, axis_tick_size - 2), color="#FF6B6B"))

st.plotly_chart(fig, use_container_width=fit_width, theme=None)

if show_critical_path and critical_ids:
    st.caption(f"🔴 Critical path: {' → '.join(t for t in leaf_order if t in critical_ids)}")

# --------------------------------------------------------- Export for PowerPoint ---------------------------------------------------------
st.subheader("3. Export")

# Export canvases match PowerPoint slide/placeholder proportions exactly. A slide-shaped
# image dropped onto a slide never needs non-uniform scaling, so fonts can't stretch.
SLIDE_SIZES = {
    "Full slide — Widescreen 16:9": (1920, 1080),
    "Full slide — Standard 4:3": (1600, 1200),
    "Half slide (content area under a title)": (1920, 780),
}


def build_ppt_figure(base_fig, title_size, tick_size, label_size, canvas_w, canvas_h, is_swimlane):
    """Clone the chart at exact slide proportions with legibility floors on every font."""
    export_fig = go.Figure(base_fig)
    title_size, tick_size, label_size = max(title_size, 18), max(tick_size, 14), max(label_size, 13)
    export_fig.update_xaxes(tickfont=dict(size=tick_size), title_font=dict(size=title_size), automargin=True)
    export_fig.update_yaxes(tickfont=dict(size=label_size), title_font=dict(size=title_size), automargin=True)
    export_fig.update_layout(
        width=canvas_w, height=canvas_h,
        margin=dict(l=160 if is_swimlane else 40, r=60, t=90, b=80),
        legend=dict(font=dict(size=label_size), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return export_fig


col_exp1, col_exp2 = st.columns([1, 1])
with col_exp1:
    slide_format = st.selectbox("Export size (matches the slide, so fonts never stretch)", list(SLIDE_SIZES.keys()))
    generate_clicked = st.button("🖼️ Prepare PowerPoint-ready PNG", use_container_width=True)
if generate_clicked:
    try:
        exp_w, exp_h = SLIDE_SIZES[slide_format]
        ppt_fig = build_ppt_figure(fig, axis_title_size, axis_tick_size, task_label_size,
                                    exp_w, exp_h, group_mode == "Swimlanes by Resource")
        st.session_state['ppt_png'] = ppt_fig.to_image(format="png", scale=2)
        st.session_state['ppt_png_name'] = f"gantt_chart_{exp_w}x{exp_h}.png"
    except Exception as e:
        st.error(f"Couldn't generate the PNG. Make sure the 'kaleido' package is installed "
                 f"(`pip install -U kaleido`). Details: {e}")

if 'ppt_png' in st.session_state:
    with col_exp2:
        st.download_button("⬇️ Download PPT PNG", st.session_state['ppt_png'],
                            st.session_state.get('ppt_png_name', 'gantt_chart_ppt.png'),
                            "image/png", use_container_width=True)
    st.caption("💡 In PowerPoint, insert the image and drag only the **corner** handles (or size it full-slide). "
               "The image already matches the slide's shape — side handles are what stretch fonts. "
               "For very long task lists, reduce 'Row height' or export two charts instead of squeezing one.")
