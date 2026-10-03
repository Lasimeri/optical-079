// KWin script: the tracker window (caption laptopcam, or any mpv) to the
// bottom-left of its screen, 360x270, above other windows, on every desktop.
var ws = workspace.windowList ? workspace.windowList() : workspace.clientList();
for (var i = 0; i < ws.length; i++) {
    var w = ws[i];
    if ((w.caption && /laptopcam|bedroomcam/.test(w.caption)) || (w.resourceClass && /mpv/.test(w.resourceClass))) {
        var g = w.output ? w.output.geometry : workspace.virtualScreenGeometry;
        w.frameGeometry = {x: g.x + 24, y: g.y + g.height - 300, width: 360, height: 270};
        w.keepAbove = true; w.onAllDesktops = true;
    }
}
