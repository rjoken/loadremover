import sys, getopt, cv2, ffmpegcv, time, tqdm
import numpy as np

def main(argv):
    if(len(argv) != 2):
        print("Invalid number of arguments. Usage is: loadremover.py inputfile outputfile")
        sys.exit()
    
    infile = argv[0]
    outfile = argv[1]
    
    # can be changed if not properly detecting
    threshold = .8
    
    # template filename
    template = cv2.imread("loading.png")
    
    # get height and width of template image
    h, w = template.shape[:-1]
    
    cap = cv2.VideoCapture(infile)
    out = cv2.VideoWriter(outfile,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(5),
    (int(cap.get(3)), int(cap.get(4))))
    # 5 = fps, 3 = width, 4 = height
    
    # tqdm for progress bar
    # cap.get(7) = frame count
    for _ in tqdm.trange(int(cap.get(7))):
        ret, img = cap.read()
        
        # IF NO FRAMES LEFT, BREAK
        if not ret:
            break

        # match frame to template
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        
        # IF NO MATCH, WRITE FRAME
        if(not (res >= threshold).any()):
            out.write(img)
    
    # done with file
    out.release()
       
if __name__ == "__main__":
	main(sys.argv[1:])