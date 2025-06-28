import sys, cv2
import numpy as np

def main(argv):
    if(len(argv) != 2):
        print("Invalid number of arguments. Usage is: loadremover.py inputfile outputfile")
        sys.exit()
    
    infile = argv[0]
    outfile = argv[1]
    threshold = .9
    
    template = cv2.imread("loading_img.png")
    img_rgb = cv2.imread(infile)
    h, w = template.shape[:-1]
    
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        
    cv2.imwrite(outfile, img_rgb)
       
if __name__ == "__main__":
	main(sys.argv[1:])