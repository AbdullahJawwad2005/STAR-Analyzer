import h5py
import numpy as np


def load_sleap(path):
    """Load and validate a SLEAP analysis HDF5 file."""
    with h5py.File(path, "r") as f:
        required = ("tracks", "node_names", "track_names")
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"Missing required SLEAP dataset(s): {', '.join(missing)}")

        tracks = f["tracks"][:]
        node_names = [_decode_name(n) for n in f["node_names"][:]]
        track_names = [_decode_name(n) for n in f["track_names"][:]]
        edge_inds = (
            f["edge_inds"][:]
            if "edge_inds" in f
            else np.empty((0, 2), dtype=int)
        )
        frame_idx = (
            f["frame_idx"][:]
            if "frame_idx" in f
            else np.arange(tracks.shape[0])
        )

    if tracks.ndim != 4 or tracks.shape[1] != 2:
        raise ValueError(
            "Expected tracks shape (n_frames, 2, n_nodes, n_tracks) or "
            "(n_tracks, 2, n_nodes, n_frames); "
            f"got {tracks.shape}."
        )

    # Detect (n_tracks, 2, n_nodes, n_frames) layout and transpose to
    # (n_frames, 2, n_nodes, n_tracks).
    # Do NOT reset frame_idx here — the loaded frame_idx already has n_frames
    # elements with the correct video-frame numbers from the H5 file.
    # Resetting it to np.arange() would break sparse exports where frame_idx
    # contains non-contiguous video frame numbers (e.g. [0, 5, 10, 20, ...]).
    if len(track_names) == tracks.shape[0] and len(node_names) == tracks.shape[2]:
        tracks = tracks.transpose(3, 1, 2, 0)

    if len(node_names) != tracks.shape[2]:
        raise ValueError("Node name count does not match tracks node dimension.")
    if len(track_names) != tracks.shape[3]:
        raise ValueError("Track name count does not match tracks track dimension.")
    if len(frame_idx) != tracks.shape[0]:
        raise ValueError("frame_idx length does not match number of tracked frames.")

    return {
        "tracks": tracks,
        "node_names": node_names,
        "track_names": track_names,
        "edge_inds": edge_inds,
        "frame_map": {int(fi): i for i, fi in enumerate(frame_idx)},
        "nan_count": int(np.isnan(tracks).sum()),
    }


def _decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
