#!/bin/bash
# laptop-media-status.sh NAME: one line on whether this laptop plays sound its
# echo canceller does not see (a browser video straight to HDMI), for
# lapcams.sh's status pull (copied to ~/.cache/lapcam/media.sh on the laptop):
#   t=UNIX_US media=NAME playing=N apps=A,B
# Counted: uncorked streams with an application name, not the desktop's
# pw-cat links, not on the echo canceller's sink (*_aec_ref). Firefox corks
# its stream when the video pauses or ends (seen on the E16, 2026-10-03).
export XDG_RUNTIME_DIR=/run/user/$(id -u)
sinks=$(pactl list short sinks 2>/dev/null | awk '{printf "%s=%s ", $1, $2}')
pactl list sink-inputs 2>/dev/null | awk -v t="$(date +%s%6N)" -v n="${1:-laptop}" -v sinks="$sinks" '
    BEGIN { k = split(sinks, a, " "); for (i = 1; i <= k; i++) { split(a[i], b, "="); name[b[1]] = b[2] } }
    function done() {
        if (id != "" && app != "" && app != "pw-cat" && c == "no" && name[s] !~ /_aec_ref$/) {
            p++; apps = apps (apps != "" ? "," : "") app
        }
    }
    /^Sink Input #/ { done(); id = $3; app = ""; c = ""; s = "" }
    /^\tSink: / { s = $2 }
    /^\tCorked: / { c = $2 }
    /application.name = / { app = $0; sub(/.*application.name = "/, "", app); sub(/".*/, "", app); gsub(/ /, "_", app) }
    END { done(); printf "t=%s media=%s playing=%d apps=%s\n", t, n, p, (apps != "" ? apps : "-") }'
