# QR Code Detection Image Processing Assignment
##### University of Auckland, COMPUTER SCIENCE, COMPSCI 373

The purpose of this program is to take in a simple PNG image file with a QR Code and perform image processing techniques to 
'detect' where the QR Code is and surround it via a Bounding Box. This bounding box is then used to cut out that part of the image to 
decode the QR Code using pyzbar. 

- This program was an assignment to test our image processing knowledge. This program only works for a small amount of simple QR Codes.
- Improving this program to work for most QR Codes in any image goes far beyond the knowledge taught in class.
- As Runtime was not marked, I did not focus on optimzation so expect long runtimes for some of the images. 
- Image processing techniques are done step by step and is step dependent, therefore, significant optimizations are quite   difficult to achieve
and I don't want to break any code. 

##### Note
- To change which image to process, just change the filename in first line of main() function. Image files are located in the images folder
- Only PNG files are allowed to be processed 
- You can see how image processing techniques manipulate the input image to the output result via the PNG files that are generated when you run 
the program (called Step1.png, Step2.png...and so on)
- I don't own any images provided in this repository/project

##### Algorithm Explanation:
Step 1: Read input image, convert RGB data to greyscale and stretch values to lie between 0 and 255
Step 2: Compute Horizontal Edges using 3x3 sobel filter masks
Step 3: Compute Vertical Edges using 3x3 sobel filter masks
Step 4: Compute Edge (gradient) magnitude 
Step 5: Smooth over the Edge magnitude using box averaging mean filter and stretch contrast to 0 and 255
Step 6: Perform a thresholding operation to get the edge regions as a binary image
Step 7: Perform morphological closing operation to fill holes 
Step 8: Perform a connected component analysis to find the largest/correct connected object 
Step 9: Extract Bounding Box around correct region 
