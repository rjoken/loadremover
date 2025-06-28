import sys, argparse, tqdm, cv2
from concurrent.futures import ThreadPoolExecutor

def parse_timecode(tc: str) -> float:
    """
    Parse a timecode string "HH:MM:SS.mmm" into total seconds.
    """
    parts = tc.split(':')
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
    elif len(parts) == 2:
        h = 0
        m = int(parts[0])
        s = float(parts[1])
    else:
        raise argparse.ArgumentTypeError(f"Invalid timecode format: '{tc}', expected HH:MM:SS.mmm or MM:SS.mmm")
    return h * 3600 + m * 60 + s
    
def format_timecode(seconds: float) -> str:
    """
    Format seconds into a timecode string "HH:MM:SS.mmm".
    """
    total_ms = int(round(seconds * 1000))
    hrs = total_ms // 3600000
    total_ms %= 3600000
    mins = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
    
def match_frame(frame, template, threshold):
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    return not (res >= threshold).any() # True = write frame 

def process_batch(frames, template, threshold):
    result = []
    for idx, frame in frames:
        keep = match_frame(frame, template, threshold)
        if keep:
            result.append((idx, frame))
    return result
    
def main(argv):
    parser = argparse.ArgumentParser(description="Remove loading frames from a segment of video and report runtime")
    parser.add_argument('infile', help="Path to input video file")
    parser.add_argument('outfile', help="Path to output video file")
    parser.add_argument('--start', type=parse_timecode, default=None, help="Start timecode (HH:MM:SS.mmm)")
    parser.add_argument('--end', type=parse_timecode, default=None, help="End timecode (HH:MM:SS.mmm)")
    parser.add_argument('--threshold', type=float, default=0.9, help="Match threshold for template removal")
    parser.add_argument('--workers', type=int, default=1, help="Number of worker threads")
    parser.add_argument('--batch', type=int, default=64, help="Batch size for frame processing")
    parser.add_argument('--comparison', type=str, default="loading.png", help="Path to comparison image file")
    args = parser.parse_args()
    
    start_ms = args.start * 1000.0 if args.start is not None else None
    end_ms   = args.end   * 1000.0 if args.end   is not None else None
    
    template = cv2.imread(args.comparison)
    if template is None:
        print(f"Could not load template image. '{args.comparison}'")
        sys.exit(1)
    
    cap = cv2.VideoCapture(args.infile)
    if not cap.isOpened():
        print(f"Could not open input video. '{args.infile}'")
        sys.exit(1)
        
    if start_ms is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(args.outfile, fourcc, fps, (width, height))
    
    total_written = 0
    progress_bar = tqdm.tqdm(total=total_frames)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            batch, timestamps = [], []
            for _ in range(args.batch):
                ret, frame = cap.read()
                if not ret:
                    break
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                # if end time given, stop after it
                if end_ms is not None and ts > end_ms:
                    break
                batch.append(frame)
                timestamps.append(ts)

            if not batch:
                break

            # Filter out loading frames
            keep_flags = list(executor.map(
                lambda f: match_frame(f, template, args.threshold),
                batch
            ))
            for frame, keep in zip(batch, keep_flags):
                if keep:
                    out.write(frame)
                    total_written += 1
                progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()
    
    duration = total_written / fps
    penalty_duration_york = duration + 38
    penalty_duration_lanc = duration + 77
    print(f"Output duration: {format_timecode(duration)}")
    print(f"Output duration with penalty starting Lancastrians: {format_timecode(penalty_duration_lanc)}")
    print(f"Output duration with penalty starting Yorkists: {format_timecode(penalty_duration_york)}")
    
if __name__ == "__main__":
    main(sys.argv[1:])