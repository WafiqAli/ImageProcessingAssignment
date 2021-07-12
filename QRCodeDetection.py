"""
QR Code Detection Image Processing Assignment for University of Auckland, COMPUTER SCIENCE, COMPSCI 373
Author: Wafiq Ali

The purpose of this program is to take in a simple PNG image file with a QR Code and perform image processing techniques to 
'detect' where the QR Code is and surround it via a Bounding Box. This bounding box is then used to cut out that part of the image to 
decode the QR Code using pyzbar. 

-This program was an assignment to test our image processing knowledge. This program only works for a small amount of simple QR Codes.
-Improving this program to work for most QR Codes in any image goes far beyond the knowledge taught in class.
-As Runtime was not marked, I did not focus on optimzation so expect long runtimes for some of the images. 
-Image processing techniques are done step by step and is step dependent, therefore, significant optimizations are quite difficult to achieve
and I don't want to break any code. 

Note:
-To change which image to process, just change the filename in first line of main() function. Image files are located in the images folder
-Only PNG files are allowed to be processed 
-You can see how image processing techniques manipulate the input image to the output result via the PNG files that are generated when you run 
the program (called Step1.png, Step2.png...and so on)

Algorithm Explanation:
Step 1: Read input image, convert RGB data to greyscale and stretch values to lie between 0 and 255
Step 2: Compute Horizontal Edges using 3x3 sobel filter masks
Step 3: Compute Vertical Edges using 3x3 sobel filter masks
Step 4: Compute Edge (gradient) magnitude 
Step 5: Smooth over the Edge magnitude using box averaging mean filter and stretch contrast to 0 and 255
Step 6: Perform a thresholding operation to get the edge regions as a binary image
Step 7: Perform morphological closing operation to fill holes 
Step 8: Perform a connected component analysis to find the largest/correct connected object 
Step 9: Extract Bounding Box around correct region 

"""
from math import sqrt
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import matplotlib.path as mplPath
import numpy as np
from pyzbar.pyzbar import decode
import imageIO.png

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            g = 0.299 * (pixel_array_r[i][j]) + 0.587 * (pixel_array_g[i][j]) + 0.114 * (pixel_array_b[i][j])
            greyscale_pixel_array[i][j] = round(g)
    
    
    return greyscale_pixel_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height): # Contrast Stretching Function
    
    contrast_stretched_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    gmax = 255
    gmin = 0
    
    flat_list = []
    for i in range(image_height):
        for j in range(image_width):
            flat_list.append(pixel_array[i][j])
            
    fHigh = max(flat_list)
    fLow = min(flat_list)
    
    for i in range(image_height):
        for j in range(image_width):
            try:
                sOut =  (pixel_array[i][j] - fLow)*((gmax - gmin)/ (fHigh - fLow)) + gmin
            
            except ZeroDivisionError: 
                sOut = 0
            
            if sOut < gmin:
                sOut = gmin
            elif sOut > gmax:
                sOut = gmax
            
            sOut = round(sOut)
            contrast_stretched_pixel_array[i][j] = sOut
            
    return contrast_stretched_pixel_array 


def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    
    result_list = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_kernel =  [[-1/8, 0, 1/8], [-1/4, 0, 1/4], [-1/8, 0, 1/8]]
    
    
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            
            temp_list = []
            count_col = 0
            count_row = 0
            for a in range(i - 1, i + 2):
                count_col = 0
                for b in range(j - 1, j + 2):
                    temp_list.append(pixel_array[a][b] * sobel_kernel[count_row][count_col])
                    
                    count_col += 1
                count_row += 1
            
            result_list[i][j] = round(abs(sum(temp_list)))
    
    
    return result_list


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
  
    
    result_list = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_kernel =  [[1/8, 1/4, 1/8], [0, 0, 0], [-1/8, -1/4, -1/8]]
    
    
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            
            temp_list = []
            count_col = 0
            count_row = 0
            for a in range(i - 1, i + 2):
                count_col = 0
                for b in range(j - 1, j + 2):
                    temp_list.append(pixel_array[a][b] * sobel_kernel[count_row][count_col])
                    
                    count_col += 1
                count_row += 1
            
            result_list[i][j] = round(abs(sum(temp_list)))
    
    
    return result_list


def computeEdgeMagnitude(horizontal_edges, vertical_edges, image_width, image_height):

    edge_mag_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            result_pixel = sqrt(pow(horizontal_edges[i][j], 2) + pow(vertical_edges[i][j], 2))
            edge_mag_array[i][j] = round(result_pixel)

    return edge_mag_array


def computeBoxAveraging5x5(pixel_array, image_width, image_height): # Mean Filter (THIS IS THE ONE I WILL USE. NOT THE GAUSSIAN FILTER)
    
    result_list = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(0, image_height):
        for j in range(0, image_width):
            
            sum = 0
           
            for a in range(i - 2, i + 3):
                for b in range(j - 2, j + 3):
                    try:
                        sum += pixel_array[a][b]
                    except IndexError:
                        pass
            
            result_list[i][j] = round(sum / 25)
    
    
    return result_list


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    
    result_list = createInitializedGreyscalePixelArray(image_width, image_height)
    filter_kernel =  [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]
    
    pixel_array_repeat = createInitializedGreyscalePixelArray(image_width + 2, image_height + 2)
    
    pixel_array_repeat[0][0] = pixel_array[0][0]
    pixel_array_repeat[0][image_width + 1] = pixel_array[0][image_width - 1]
    pixel_array_repeat[image_height + 1][0] = pixel_array[image_height - 1][0]
    pixel_array_repeat[image_height + 1][image_width + 1] = pixel_array[image_height - 1][image_width - 1]
    
    for i in range(image_width):
        pixel_array_repeat[0][i + 1] = pixel_array[0][i]
        pixel_array_repeat[image_height + 1][i + 1] = pixel_array[image_height - 1][i]
    
    for j in range(image_height):
        pixel_array_repeat[j + 1][0] = pixel_array[j][0]
        pixel_array_repeat[j + 1][image_width + 1] = pixel_array[j][image_width - 1]
    
    for i in range(image_height):
        for j in range(image_width):
            pixel_array_repeat[i + 1][j + 1] = pixel_array[i][j]
    
    
    i_repeat = 1
    j_repeat = 1
    for i in range(image_height):
        j_repeat = 1
        for j in range(image_width):
            
            temp_list = []
            count_col = 0
            count_row = 0
            for a in range(i_repeat - 1, i_repeat + 2):
                count_col = 0
                for b in range(j_repeat - 1, j_repeat + 2):
                    
                    temp_list.append(pixel_array_repeat[a][b] * filter_kernel[count_row][count_col])
                    
                    count_col += 1
                count_row += 1
            
            result_list[i][j] = round(sum(temp_list))
            j_repeat += 1
        i_repeat += 1
        
            
    return result_list


def smoothNTimesAndContrastStretch(n, px_array, image_width, image_height): # Uses filter on image several times, then applies contrast stretching
    
    stretched_smoothed_px_array = px_array
    
    for i in range(n):
        #stretched_smoothed_px_array = computeGaussianAveraging3x3RepeatBorder(stretched_smoothed_px_array, image_width, image_height)
        stretched_smoothed_px_array = computeBoxAveraging5x5(stretched_smoothed_px_array, image_width, image_height)


    stretched_smoothed_px_array = scaleTo0And255AndQuantize(stretched_smoothed_px_array, image_width, image_height) 
    return stretched_smoothed_px_array


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            
            if pixel_array[i][j] >= threshold_value:
                result_array[i][j] = 255
    
    
    return result_array
    

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    #SE = 3x3 8 neighbourhood. element = 1
    
    for i in range(image_height):
        for j in range(image_width):
            
            not_fit = False
            for a in range(i - 1, i + 2):
                for b in range(j - 1, j + 2):
                    try:
                        if pixel_array[a][b] == 0:
                            not_fit = True
                    except IndexError:
                        not_fit = True
            if not_fit == True:
                result_array[i][j] = 0
            else:
                result_array[i][j] = 255
    
    
    
    return result_array


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            
            hit = False
            for a in range(i - 1, i + 2):
                for b in range(j - 1, j + 2):
                    try:
                        if pixel_array[a][b] != 0:
                            hit = True
                    except IndexError:
                        pass
    
            if hit == True:
                result_array[i][j] = 255
            else:
                result_array[i][j] = 0
    
    
    return result_array


def morphologicalClosing(n, pixel_array, image_width, image_height):  # Dilations followed by Erosions <-- Closing
    
    processing_array = pixel_array
    for i in range(n):
        processing_array = computeDilation8Nbh3x3FlatSE(processing_array, image_width, image_height)
        processing_array = computeErosion8Nbh3x3FlatSE(processing_array, image_width, image_height)

    return processing_array


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    mapping_dict = {}
    visited_pixels = createInitializedGreyscalePixelArray(image_width, image_height)
    current_label = 1
    
    for i in range(image_height):
        for j in range(image_width):
            
            if (visited_pixels[i][j] == 0 and pixel_array[i][j]):
                queue = Queue()
                visited_pixels[i][j] = 1
                
                mapping_dict[current_label] = 0
                
                queue.enqueue((i,j))
                while queue.isEmpty() != True:
                    current_pixel = queue.dequeue()
                    i_prime = current_pixel[0]
                    j_prime = current_pixel[1]
                    result_array[i_prime][j_prime] = current_label
                    mapping_dict[current_label] += 1
                    
                    if (j_prime - 1 >= 0):
                        if (visited_pixels[i_prime][j_prime - 1] == 0 and pixel_array[i_prime][j_prime - 1]):
                            visited_pixels[i_prime][j_prime - 1] = 1
                            queue.enqueue((i_prime, j_prime - 1))
                            
                    if (j_prime + 1 < image_width):
                        if (visited_pixels[i_prime][j_prime + 1] == 0 and pixel_array[i_prime][j_prime + 1]):
                            visited_pixels[i_prime][j_prime + 1] = 1
                            queue.enqueue((i_prime, j_prime + 1))
                    
                    if (i_prime - 1 >= 0):
                        if (visited_pixels[i_prime - 1][j_prime] == 0 and pixel_array[i_prime - 1][j_prime]):
                            visited_pixels[i_prime - 1][j_prime] = 1
                            queue.enqueue((i_prime - 1, j_prime))
                            
                    if (i_prime + 1 < image_height):
                        if (visited_pixels[i_prime + 1][j_prime] == 0 and pixel_array[i_prime + 1][j_prime]):
                            visited_pixels[i_prime + 1][j_prime] = 1
                            queue.enqueue((i_prime + 1, j_prime))
                
                current_label += 1
                
    return (result_array, mapping_dict)


def returnLargestObjectAfterCCL(pixel_array, mapping_dict, image_width, image_height): 
    
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    largestObj_ID = max(mapping_dict, key=mapping_dict.get)
   
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] == largestObj_ID:
                result_array[i][j] = 255
    
    return result_array, mapping_dict, largestObj_ID


def getBoundaryBoxValues(pixel_array, image_width, image_height): # Computes values required to construct rectangle over QR Code
    minX = [-1,-1]
    minY = [-1,-1]
    maxX = [-1,-1]
    maxY = [-1,-1]

    for i in range(image_height):
        for j in range(image_width):

            if pixel_array[i][j] != 0:
                if minX[0] == -1: #initialises minX pixel location
                    minX[0] = j
                    minX[1] = i
                if minY[1] == -1: #initialises minY pixel location
                    minY[0] = j
                    minY[1] = i
                
                if j < minX[0]: 
                    minX[0] = j
                    minX[1] = i
                
                if i < minY[1]:
                    minY[0] = j
                    minY[1] = i
                
                if j > maxX[0]:
                    maxX[0] = j
                    maxX[1] = i

                if i > maxY[1]:
                    maxY[0] = j
                    maxY[1] = i


    return tuple(minX), tuple(minY), tuple(maxX), tuple(maxY)


def FindCorrectObjectAndBoundaryValues(pixel_array, mapping_dict, image_width, image_height): # Finds correct object and returns corner values for bounding box
    
    correctObj = False
    #original_mapping_dict = mapping_dict

    while(correctObj != True):
        
        if not mapping_dict:
            break

        binary_img_largestObj, mapping_dict, largestObj_ID = returnLargestObjectAfterCCL(pixel_array, mapping_dict, image_width, image_height)
        minX, minY, maxX, maxY = getBoundaryBoxValues(binary_img_largestObj, image_width, image_height)

        TL, TR, BL, BR = [minX[0], minY[1]],  [maxX[0], minY[1]],  [minX[0], maxY[1]],  [maxX[0], maxY[1]]

        polygonPath = mplPath.Path([minY, maxX, maxY, minX])
        rectanglePath = mplPath.Path([TL, TR, BR, BL])

        print("For PolygonPath : ", minX, minY, maxX, maxY)
        print("For RectanglePath : ", TL, TR, BL, BR)

        within = 0
        zeroPixelCount = 0
        for i in range(image_height):
            for j in range(image_width):

                if binary_img_largestObj[i][j] != 0:
                    if polygonPath.contains_point((j, i)):
                        within += 1
                else:
                    if polygonPath.contains_point((j, i)):         
                        zeroPixelCount += 1
        
        percent_within = (within / mapping_dict[largestObj_ID])
        
        try:
            percent_zeroPixels = (zeroPixelCount / (zeroPixelCount + within))
        except ZeroDivisionError:
            percent_zeroPixels = 0

        print("Polygon Percentage: " , percent_within)
        print("Zero Pixel Count: ", percent_zeroPixels)
        if percent_within >= 0.9 and percent_zeroPixels <= 0.3:  
            correctObj = True
            C1 = minY
            C2 = maxX
            C3 = maxY
            C4 = minX
        
        else:
            within = 0 
            zeroPixelCount = 0
            for i in range(image_height):
                for j in range(image_width):

                    if binary_img_largestObj[i][j] != 0:
                        if rectanglePath.contains_point((j, i)):
                            within += 1 
                    else:
                        if polygonPath.contains_point((j, i)):           
                            zeroPixelCount += 1

            percent_within = (within / mapping_dict[largestObj_ID])
            
            try:
                percent_zeroPixels = (zeroPixelCount / (zeroPixelCount + within))
            except ZeroDivisionError:
                percent_zeroPixels = 0  
            
            print("Rectangle Percentage: " , percent_within)
            print("Zero Pixel Count: ", percent_zeroPixels)
            if percent_within >= 0.9 and percent_zeroPixels <= 0.3:  
                correctObj = True
                C1 = TL
                C2 = TR
                C3 = BR
                C4 = BL
            else:
                del mapping_dict[largestObj_ID]


    if correctObj != True:
        return None, -1, -1, -1, -1

    
    return binary_img_largestObj, C1, C2, C3, C4 
    

def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    

# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename+".png", 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()



def getQRCodeFromImage(pixel_array, bounding_box, image_width, image_height): # Cuts out the QR code from original image and returns this as an unsigned 8-bit Numpy array
    
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)


    for i in range(image_height):
        for j in range(image_width):

            if pixel_array[i][j] != 0:
                if bounding_box.contains_point((j, i)):
                    result_array[i][j] = pixel_array[i][j]
           
    result = np.array(result_array, dtype='uint8')
    return result

def decodeQR(image_array, image_width, image_height): # Decodes QR code using pyzbar.decode()

    decoded = decode((image_array.tobytes(), image_width, image_height))
    return decoded

def main():
    filename = "./images/poster1small.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)

    # Step 1: Converting RGB to greyscale and stretching values to be between 0 and 255
    greyscale_px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    contrast_stretched_px_array = scaleTo0And255AndQuantize(greyscale_px_array, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step1", contrast_stretched_px_array, image_width, image_height)
    
    # Step 2: Compute Horizontal Edges 
    horizontal_edges = computeHorizontalEdgesSobelAbsolute(contrast_stretched_px_array, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step2", horizontal_edges, image_width, image_height)
    
    # Step 3: Compute Vertical Edges
    vertical_edges = computeVerticalEdgesSobelAbsolute(contrast_stretched_px_array, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step3", vertical_edges, image_width, image_height)
    
    # Step 4: Compute Edge (gradient) magnitude
    edge_grad_mag = computeEdgeMagnitude(horizontal_edges, vertical_edges, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step4", edge_grad_mag, image_width, image_height)
    
    # Step 5: Smooth over the Edge using Mean averaging filter and stretch
    smoothed_stretched_px_array = smoothNTimesAndContrastStretch(8, edge_grad_mag, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step5", smoothed_stretched_px_array, image_width, image_height)
    
    # Step 6: Perform a threshold operation to get edge regions as binary image
    thresh_val = 70
    binary_img = computeThresholdGE(smoothed_stretched_px_array, thresh_val, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step6", binary_img, image_width, image_height)
    
    # Step 7 (Optional): Perform morphological closing operations to fill holes -> dilations followed by erosions
    steps = 8
    binary_img_closed = morphologicalClosing(steps, binary_img, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step7", binary_img_closed, image_width, image_height)
    
    # Step 8 AND Step 9: Perform Connected Component Analysis (CCL) to find largest object and Find Bounding Box Values
    (CCL_img, mapping_dict) = computeConnectedComponentLabeling(binary_img_closed, image_width, image_height)
    binary_img_correctObj, C1, C2, C3, C4 = FindCorrectObjectAndBoundaryValues(CCL_img,mapping_dict, image_width, image_height)
    writeGreyscalePixelArraytoPNG("Step8", binary_img_correctObj, image_width, image_height)
    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))
    
    #pyplot.imshow(contrast_stretched_px_array, cmap="gray")
    #pyplot.imshow(horizontal_edges, cmap="gray")
    #pyplot.imshow(vertical_edges, cmap="gray")
    #pyplot.imshow(edge_grad_mag, cmap="gray")
    #pyplot.imshow(smoothed_stretched_px_array, cmap="gray")
    #pyplot.imshow(binary_img, cmap="gray")
    #pyplot.imshow(binary_img_closed, cmap="gray")
    #pyplot.imshow(binary_img_correctObj, cmap="gray")

    rect_polygon = Polygon([C1, C2, C3, C4], closed=True, linewidth=3, edgecolor='g', facecolor='none')
    QRCode_array = getQRCodeFromImage(greyscale_px_array, rect_polygon, image_width, image_height)

    decoded = decodeQR(QRCode_array, image_width, image_height)
    
    if (len(decoded) == 0):
        print("ERROR: Not able to Decode QRCode...")
        print("Reasons: 1. Bounding Box Cutting off parts (occurs often...) , 2. Not calculating correct object (Step 8 and 9)") 
    else:
        print("Decoded QRCODE: ")

    print(decoded)


# get access to the current pyplot figure
    axes = pyplot.gca()
    # create a rectangle bounding box with size almost identical to QR Code that starts at top left of QR Code with a line width of 3
    #rect = Rectangle( (332, 128), 470 - 332, 266 - 128, linewidth=3, edgecolor='g', facecolor='none' )
    # paint the rectangle over the current plot
    axes.add_patch(rect_polygon)

    # plot the current figure
    pyplot.show()



if __name__ == "__main__":
    main()