"""CSS for the TrajViz replica: cards, chips and the step browser."""

CSS = """
:root { --tv-border: #e5e7eb; --tv-muted: #6b7280; --tv-card: #ffffff; --tv-good: #059669;
        --tv-warn: #d97706; --tv-bad: #dc2626; --tv-accent: #2563eb; --tv-soft: #f9fafb; }
.tv-header { display: flex; align-items: baseline; gap: 12px; margin: 6px 0 10px; }
.tv-header .tv-title { font-size: 22px; font-weight: 700; }
.tv-header .tv-subtitle { font-size: 13px; color: var(--tv-muted); }
.tv-banner { font-size: 13px; padding: 6px 2px; color: #374151; }
.tv-banner.tv-error { color: var(--tv-bad); }
.tv-warnings { font-size: 12px; color: var(--tv-warn); margin: 2px 0 6px; }

.tv-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin: 14px 0; }
.tv-kpi { background: var(--tv-card); border: 1px solid var(--tv-border); border-left: 4px solid var(--tv-border);
          border-radius: 10px; padding: 12px 14px; min-height: 120px; }
.tv-kpi.good { border-left-color: var(--tv-good); } .tv-kpi.warn { border-left-color: var(--tv-warn); }
.tv-kpi.bad { border-left-color: var(--tv-bad); }
.tv-kpi-label { font-size: 11px; font-weight: 700; letter-spacing: .06em; color: #374151; text-transform: uppercase;
                border-bottom: 1px solid var(--tv-border); padding-bottom: 4px; margin-bottom: 6px; }
.tv-kpi-value { font-size: 24px; font-weight: 700; }
.tv-kpi-sub { font-size: 12px; color: #374151; margin-top: 2px; }
.tv-kpi-note { font-size: 11px; margin-top: 4px; }
.tv-kpi-note.good { color: var(--tv-good); } .tv-kpi-note.warn { color: var(--tv-warn); }
.tv-kpi-note.bad { color: var(--tv-bad); }
.tv-kpi-row { display: flex; justify-content: space-between; font-size: 11px; color: #374151; margin-top: 3px; }
.tv-spark { margin-top: 6px; }

.tv-section-subtitle { font-size: 13px; font-style: italic; color: var(--tv-muted); margin: 4px 0 10px; }
.tv-issues { border: 1px solid var(--tv-border); border-radius: 8px; padding: 6px 12px; background: var(--tv-card); }
.tv-issues summary { cursor: pointer; font-weight: 600; font-size: 14px; }
.tv-issues .tv-issues-hint { font-size: 12px; color: var(--tv-muted); margin: 4px 0 8px; }
.tv-issue { border-left: 3px solid var(--tv-warn); padding: 6px 12px; margin: 8px 0; }
.tv-issue.error { border-left-color: var(--tv-bad); } .tv-issue.bottleneck { border-left-color: var(--tv-accent); }
.tv-issue-title { font-weight: 600; font-size: 14px; }
.tv-issue-detail { font-size: 12px; color: #374151; margin-top: 2px; }
.tv-issue-why { font-size: 12px; color: var(--tv-muted); margin-top: 4px; }
.tv-issue-fix { font-size: 12px; margin-top: 6px; padding: 6px 8px; background: #ecfdf5; border-radius: 6px; }
.tv-chip { display: inline-block; padding: 1px 7px; margin: 2px 3px 2px 0; border-radius: 10px; font-size: 11px;
           background: #eef2ff; color: #3730a3; font-family: ui-monospace, monospace; }
.tv-details-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tv-detail-chip { display: inline-flex; flex-direction: column; background: var(--tv-card); border: 1px solid var(--tv-border);
                  border-radius: 8px; padding: 6px 10px; min-width: 100px; }
.tv-detail-chip span:first-child { font-size: 10px; color: var(--tv-muted); text-transform: uppercase; }
.tv-detail-chip span:last-child { font-size: 13px; font-weight: 500; }

.tv-agent-cards { display: flex; flex-wrap: wrap; gap: 12px; }
.tv-agent-card { border: 1px solid var(--tv-border); border-radius: 8px; padding: 10px 14px; min-width: 340px; }
.tv-agent-card-head { font-size: 14px; font-weight: 600; } .tv-agent-card-head span { font-weight: 400; font-size: 12px;
                      color: var(--tv-muted); margin-left: 6px; }
.tv-agent-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 18px; margin-top: 8px; }
.tv-agent-grid div b { display: block; font-size: 15px; } .tv-agent-grid div span { font-size: 10px; color: var(--tv-muted);
                       text-transform: uppercase; letter-spacing: .05em; }

.tv-context { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; margin: 6px 0; }
.tv-context b { font-size: 16px; }
.tv-usage-bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin: 6px 0; background: #f3f4f6; }
.tv-usage-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; }
.tv-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }

.tv-card-list { max-height: 720px; overflow-y: auto; padding-right: 4px; }
.tv-step-card { border: 1px solid var(--tv-border); border-left: 4px solid #9ca3af; border-radius: 6px;
                padding: 6px 10px; margin: 6px 0; background: var(--tv-card); }
.tv-step-card.user { border-left-color: #1e40af; } .tv-step-card.tool_calls { border-left-color: #2563eb; }
.tv-step-card.tool_error { border-left-color: var(--tv-bad); background: #fef2f2; }
.tv-step-card.system_error { border-left-color: var(--tv-warn); background: #fffbeb; }
.tv-step-card.final { border-left-color: var(--tv-good); } .tv-step-card.reasoning { border-left-color: #8b5cf6; }
.tv-step-head { font-size: 12px; color: var(--tv-muted); display: flex; gap: 10px; flex-wrap: wrap; }
.tv-step-head b { color: #111827; }
.tv-step-text { font-size: 13px; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tv-step-size { font-size: 12px; color: #374151; margin-top: 2px; }
.tv-badge-sm { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 700;
               color: #fff; background: #92400e; letter-spacing: .03em; }
.tv-badge-sm.num { background: #1e3a8a; } .tv-badge-sm.user { background: #1e40af; }
.tv-badge-sm.category { background: transparent; color: #b45309; border: 1px solid #f59e0b; }
.tv-badge-sm.category.tool_error, .tv-badge-sm.category.system_error { color: var(--tv-bad); border-color: var(--tv-bad); }
.tv-badge-sm.agent { background: #e0e7ff; color: #3730a3; }
.tv-step-card.assistant { background: #fffbeb; border-color: #f59e0b; }
.tv-step-card.user { background: #eff6ff; border-color: #1e40af; }
.tv-toc { border: 1px solid var(--tv-border); border-radius: 8px; padding: 8px 10px; max-height: 720px; overflow-y: auto; }
.tv-toc-row { display: flex; gap: 8px; align-items: center; font-size: 12px; margin: 4px 0; }
.tv-detail pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; background: var(--tv-soft);
                 padding: 8px; border-radius: 6px; max-height: 360px; overflow: auto; }
.tv-detail h4 { margin: 10px 0 4px; font-size: 13px; }
.tv-muted { color: var(--tv-muted); font-size: 12px; }
.tv-table { border-collapse: collapse; font-size: 12px; width: 100%; }
.tv-table th, .tv-table td { border-bottom: 1px solid var(--tv-border); padding: 4px 8px; text-align: left; vertical-align: top; }
.tv-table th { background: var(--tv-soft); font-weight: 600; }
.tv-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 13px; color: #fff; background: var(--tv-good); }
"""
