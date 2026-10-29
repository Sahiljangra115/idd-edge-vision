# Using the detector on NOIR (optional)

NOIR (`Noir/tracker.py`, separate project) has `HumanDetector.detect(frame)` returning a list of
`Detection(label, bbox=(x1,y1,x2,y2), confidence, kind)`. Distance and steering code only read `bbox`
and `kind`, so any object with a `detect(frame)` method of that shape can replace it.

`idd/noir_adapter.py` provides `IndianRoadDetector`, an ONNX Runtime backed drop-in:

    # in NOIR, with this repo on PYTHONPATH and the ml venv
    from idd.noir_adapter import IndianRoadDetector
    from tracker import Detection          # optional: pass NOIR's own class
    human_det = IndianRoadDetector("exports/int8_minmax.onnx", classes=["person", "rider"], det_cls=Detection)

Caveats, stated plainly:

- Not integrated or tested against NOIR. NOIR was not modified. Wiring it in is a one-line swap in NOIR.
- NOIR's distance estimate assumes a known human height. Our `rider` boxes include the motorcycle,
  so keep `classes=["person"]` for distance-based following.
- NOIR is an indoor/tabletop robot; a road-trained model is a poor fit there. This is a demo hook, not a use case.
- The model is trained on pseudo labels from a different detector; see the README dataset section.
