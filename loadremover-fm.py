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
    
def match_frame(frame, threshold, roi=None):
    if roi:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    # return True to keep frame
    return mean_val > threshold
    
    
def main(argv):
    parser = argparse.ArgumentParser(description="Remove loading frames from a segment of video and report runtime")
    parser.add_argument('infile', help="Path to input video file")
    parser.add_argument('outfile', help="Path to output video file")
    parser.add_argument('--start', type=parse_timecode, default=None, help="Start timecode (HH:MM:SS.mmm)")
    parser.add_argument('--end', type=parse_timecode, default=None, help="End timecode (HH:MM:SS.mmm)")
    parser.add_argument('--workers', type=int, default=1, help="Number of worker threads")
    parser.add_argument('--batch', type=int, default=64, help="Batch size for frame processing")
    parser.add_argument('--threshold', type=float, default=10.0, help="Max mean gray-value for a frame to count as \"black\" (0=pure black, 255=white)")
    parser.add_argument('--roi', type=str, default=None, help="Region of interest as 'x,y,w,h' (pixels), only check that subarea")
    args = parser.parse_args()
    
    start_ms = args.start * 1000.0 if args.start is not None else None
    end_ms   = args.end   * 1000.0 if args.end   is not None else None
    
    cap = cv2.VideoCapture(args.infile)
    if not cap.isOpened():
        print(f"Could not open input video. '{args.infile}'")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    roi = tuple(map(int, args.roi.split(','))) if args.roi else None
    start_frame = int((args.start * fps) if args.start is not None else 0)
    end_frame = int((args.end * fps) if args.end is not None else total_frames)
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))
    frames_to_process = end_frame - start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out = cv2.VideoWriter(args.outfile, fourcc, fps, (width, height))
    
    total_written = 0
    progress_bar = tqdm.tqdm(total=frames_to_process, unit='frame')
    current_frame = start_frame 
    done = False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while current_frame < end_frame and not done:
            batch = []
            to_read = min(args.batch, end_frame - current_frame)
            for _ in range(to_read):
                ret, frame = cap.read()
                if not ret:
                    done = True
                    break
                batch.append(frame)
                current_frame += 1

            if not batch:
                break
                

            # Filter out loading frames
            keep_flags = list(executor.map(
                lambda f: match_frame(f, args.threshold, roi),
                batch
            ))
            for frame, keep in zip(batch, keep_flags):
                if keep:
                    out.write(frame)
                    total_written += 1
            progress_bar.update(len(batch))

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