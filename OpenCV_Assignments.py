'''Hi there! Here is the whole codes of OpenCV course's every single lesson which is taught in https://www.koolac.org .
 You can purchase this course from https://koolac.org/product/opencv/'''


'''01-Fundamentals'''


# 01-Pixel Concept:

'''Concept of Pixel has been explained in this video.'''





# 02-Transparency Concept:

'''The Transparency Concept and difference between Non-Transparent Image and Transparent One has been explained.'''





# 03-Color Codes:

'''RGB and HEX Codes Concept and converting them to each other has been explained.'''




# 04-imread:

'''Reading Image files with cv2 module has been explained.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')



# 05-imshow:

'''talked about imshow function for MAKING a Figure window with a given name and name of the file.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# cv2.imshow('Koolac',img) # Window name and name of the object which has been read with <<<cv2.imread>>>



# 06-waitkey:

'''When we use cv2.imshow('<window Name>',<image object created with imread>) , it wouldn't show us the picture in the output. 
we must use <<cv2.waitkey(i)>> . if "i" set to 0 AND <<cv2.waitkey(i)>> be in the last line of code, the window will be never close until we 
touch an arbitrary key. "i" can be other numbers, for example 3000. in this case, the window is up for 3 seconds, then it will be close if <<cv2.waitkey(i)>>
command be in the last line of code. If <<cv2.waitkey(i)>> wouldn't be on the last line of code , the window wouldn't be close but python interpreter
goes to the next line to interpret the code. THE Numbers are respected to MILLI-SECOND. Hence ,Entered 3000 means 3000 ms. Note that if we touch 
any arbitrary key through this interval, the window would be close and python interpreter goes to the next line to execute the rest of the code.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# cv2.imshow(winname='Koolac',mat=img) # We insert the winname as our figure name.
# cv2.waitKey(0) # Zero means waiting till Human Dooms Revelution Day!





# 07-destroyAllWindows and destroyWindow:

'''If we are going to close the image's window in middle of our programming, we use <<destroyWindow('--image_name--') or use destroyAllWindows().'''

# import cv2
#
# img1=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# img2=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\02\02.png')
# img3=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\03\03.png')
#
# cv2.imshow('Image 1',img1)
# cv2.imshow('Image 2',img2)
#
# cv2.waitKey(0)
#
# cv2.destroyWindow('Image 2')
# # cv2.destroyAllWindows()
#
# cv2.imshow('Image 3',img3)
# cv2.waitKey(0)





# 08-namedWindow:

'''What if we tend to adjust the size of opened window? Normally we cant. let's see what should we do in code.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# cv2.namedWindow('Koolac',cv2.WINDOW_NORMAL) # We use <<namedWindow()>> to make an adjustable window size, by setting its argument correspond to <<imshow()>>.
# #<cv2.WINDOW_NORMAL> will make the picture adjustable.
# cv2.imshow('Koolac',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# 09-save an image:

'''In this video , saving an image figure has been explained.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# # cv2.namedWindow('Koolac',cv2.WINDOW_NORMAL)
# cv2.imshow('Koolac',img)
# cv2.waitKey(0)
# cv2.imwrite(r"C:\Users\rezaj\PycharmProjects\Koolac\01.jpg",img) # We save it with .jpg format, instead ogf .png .
# cv2.destroyWindow('Koolac')





# 10-waitkey and ord:

'''Consider a situation where we tend to save our image when only a specific key is tabbed. (e.g.: "s" button). <<cv2.waitKey()>> return the ascii code
of the key button that has been pushed. we can use it. now let's code.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# cv2.imshow('Nimble Ant',img)
# print('If you wanna save this figure, Press "s" button. otherwise anything else:')
# key=cv2.waitKey(0)
# if key==ord('s'):         # ord() function return ascii code of an arbitrary string.
#     cv2.imwrite(r"C:\Users\rezaj\PycharmProjects\Koolac\Nimble Ant.jpg",img)
#     cv2.destroyAllWindows()
# else:
#     cv2.destroyAllWindows()





'''02-image as an array'''



# 01-BGR:

'''The RGB system color has been explained which as we know each color vast from 0 to 255. but OpenCV use BGR system color; i.e. this module
use the RGB code vice versa where we should be aware.'''



# 02-img is an array:

'''The Images where use in OpenCV are a NumPy array actually. a 3D array where. <<.shape[0]>> method indicates the number of rows,
<<.shape[1]>> indicates the number of columns and <<.shape[2]>> indicates the depth ; i.e.: BGR code of that specific color.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\05\05.png')
# cv2.imshow('Nimble Ant',img)
# print('Array Shape: ',img.shape,end='\n\n')
# print('Number of Dimension: ',img.ndim,end='\n\n')
# print('Array: ',img)
# print('If you wanna save this figure, Press "s" button. otherwise anything else:')
# key=cv2.waitKey(0)
# if key==ord('s'):         # ord() function return ascii code of an arbitrary string.
#     cv2.imwrite(r"C:\Users\rezaj\PycharmProjects\Koolac\2x2 BGR.jpg",img)
#     cv2.destroyAllWindows()
# else:
#     cv2.destroyAllWindows()





# 03-indexing and coordinate system in OpenCV:

'''As we know, images in opencv module are an numpy array. But we should note that the origin of coordination of an image in opencv , is on Top-Left.
i.e. for exapmle an array like <img> , img[0,0] is on top left and orientation of Y axis is from up to down and X axis is the default left to right.
Now consider the followed image that we are going to access to a specific pixel of it , and change its color. 
    As we see the First index belongs to Y axis and Second belongs to X axix. because in a NumPy array the first index refers to Number of row ,
     and second index refer to number of column.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\04\04.png')
# img[265:275+1,165:175+1]=[255,255,255] # We change the 5 pixel around the point [270,170] with radius of 5.
# cv2.imshow('Koolac.org',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 04-unit8 explanation:

'''NumPy arrays as an Image BGR code , are from uint8 Data type. this means <<img.dtype>> return uint8 .1- "u" means Unsigned, the values begins from
 zero, 2- "int" -> integer and 3- 8 is corresponded to 8-bit system; this is why values vast in [0,255] interval. '''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# print('Array Shape: ',img.shape)
# print('Array\'s Number of Dimension: ',img.ndim)
# print('Array\'s Data Type: ',img.dtype)





# 05-create image from array:

'''In this video we make an Image from numpy array and indicate it. first we make a whole black picture, then a whole blue one, 
then we make an image which it contains blue ,green ,red and black.'''


# import numpy as np
# import cv2
#
#
# # a Black image with 200x200 pixels:
#
# img=np.zeros([200,200,3],dtype=None) #       !!! - -  Note - -  !!! : we MUST describe dtype argument as uint8 , otherwise it consider the default value as float64
# cv2.imshow('The Black Image',img)
# cv2.imwrite(r'C:\Users\rezaj\PycharmProjects\Koolac\200x200 The Black Image.png ',img)
# cv2.waitKey(0)
#
#
# # a Blue image with 200x200 pixels:
#
# img=np.zeros([200,200,3],dtype=None) #       !!! - -  Note - -  !!! : we MUST describe dtype argument as uint8 , otherwise it consider the default value as float64
# img[:,:]=[255,0,0] # We Change all rows and columns with a channel which only contain blue spectrum
# cv2.imshow('The Blue Image',img)
# cv2.imwrite(r'C:\Users\rezaj\PycharmProjects\Koolac\200x200 The Blue Image.png ',img)
# cv2.waitKey(0)
#
#
#
# # an Image cotains Blue , Red , Green and Black with 200x200 pixels:
#
# img=np.zeros([200,200,3],dtype=None) #!!! --  Note -- !!!: We MUST describe dtype argument as uint8 , otherwise it consider the default value as float64.
# img[0:100,0:100]=[255,0,0] # We Change first hundred rows and columns with a channel which only contain blue spectrum in BGR system.
# img[0:100,100:200+1]=[0,0,255] # We Change first hundred rows and secound hundred columns with a channel which only contain red spectrum in BGR system.
# img[100:200+1,0:100]=[0,255,0] # We Change second hundred rows and first hundred columns with a channel which only contain green spectrum in BGR system.
# img[100:200+1,100:200+1]=[0,0,0] # We Change second hundred rows and second hundred columns with a channel which only contain black spectrum in BGR system.
# cv2.imshow('The BGR Image',img)
# cv2.imwrite(r'C:\Users\rezaj\PycharmProjects\Koolac\200x200 The BGR Image.png ',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 06-ROI:

'''Region of Interest (ROI) has been discussed in this video. In this video, we duplicate Koolac logo in two other coordination. let's see what happened.'''


# import numpy as np
# import cv2
#
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\10\10.png')
# height=width=25 # The Radius that we are up to gather from Koolac's logo centroid
# koolac_logo=img[100-height:100+height+1,100-width:100+width+1] # Koolac's logo central point based in [100,100] coordination.
# img[200-height:200+height+1,100-width:100+width+1]=koolac_logo # We modified coordination [200,100] by koolac_logo
# img[100-height:100+height+1,200-width:200+width+1]=koolac_logo # We modified coordination [100,200] by koolac_logo
#
# cv2.imshow('Koolac\'s Logos',img) # The final image.
# cv2.imwrite(r"C:\Users\rezaj\PycharmProjects\Koolac\Koolac's Logos.png",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()










'''03-Fundamental Operations'''



# 01-BGR to Grayscale:

'''In this video we learn how to covert a Colored BGR image into a Black&white (i.e.: Gray-Scaled) image. we also notice that the number of 
dimension in a BGR picture after converting to a gray-scaled one , drop by 1. Actually the array shape <<.shape[2]>> will be discarded and the
 array shape from (NxMxP) turn to (NxM). N and M are corresponded to number of height and width pixel, and P is the BGR channel.'''


# import cv2
# import numpy as np
#
#
# img=cv2.imread(r'C:\Users\rezaj\OneDrive\Desktop\Reza Javadzadeh.png')
#
# gray_scaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # This function is used to converting color systems
# cv2.imshow('Reza Javadzadeh B&W',gray_scaled_img)
# print('BGR Colored Image\'s Shape:  ',img.shape)
# print('Gray Scaled Image\'s Shape:  ',gray_scaled_img.shape)
# cv2.imwrite(r'C:\Users\rezaj\OneDrive\Desktop\Reza Javadzadeh BW.png',gray_scaled_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# 02-BGR to Grayscale-Complementary:

'''To check that a picture is BGR or Gray-Scaled , we should never trust to the picture itself; we should check the shape of that image as an array.
look at the following example:'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\15\15.png')
# print('Image\'s Shape: ',img.shape) # It's 3 Dimensional , so it's not Gray-Scaled.
# cv2.imshow('No. 6 (BGR)' , img)
# cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert it into a Gray-Scaled image
# cv2.imshow('No. 6 (Gray-Scaled)' , img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 03-BGR to RGB:

'''Some modules beside OpenCV, are using RGB system color map instead of BGR. In this video we learn how to convert BGR sysyem onto RGB, we may need
this conversion later.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# cv2.imshow('Nimble Ant' , img)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Convert it into a RGB System color image
# cv2.imshow('Nimble Ant (BGR 2 RGB) ' , img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 04-resize-A:

'''We learn how to resize an image . Part I'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\16\16.png')
# cv2.imshow('Milad\'s Tower' , img)
# print('Image Default\'s Size: ',img.shape)
# new_img=cv2.resize(img,(300,300)) # Resizing Image
# print('New Image\'s Size: ',new_img.shape)
# cv2.imshow('Resized Milad\'s Tower' , new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 05-resize-B:

'''If we're going resize an image with disregarding to its height and width value,(i.e. only want to multiply its height and width to arbitrary numbers)
we should act like following code:'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\15\15.png')
# new_img=cv2.resize(img,(0,0),fx=1.5,fy=0.8) # We MUST insert (0,0) tuple as input size then describe how much we want the image vasts on X-axis and Y-axis.
# cv2.imshow('First img',img)
# cv2.imshow('Second Img',new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 06-rotate:

'''In this video , we learn how to rotate our image with cv2 and SciPy.'''

# import cv2
# from scipy import ndimage # This submodule utilizes when we are going to rotate our image with a Non-conventional's number degree (i.e.: != 90,180)
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# rotate_cv2=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) # Rotating with cv2 module.
# cv2.imshow('Rotated with cv2',rotate_cv2)
# rotate_scipy=ndimage.rotate(img,angle=-40) # Rotating with scipy.ndimage submodule.
# cv2.imshow('Rotated with scipy_ndimage',rotate_scipy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 07-flip:

'''Flipping the images has been explained in this video.
        Note: if flipcode == 1 : it's Horizontal Flipping
              if flipcode == 0 : it's Vertical Flipping
              if flipcode == -1 : it's Horizontal-Vertical Flipping'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# horizontal_flipping=cv2.flip(img,flipCode=1)
# vertical_flipping=cv2.flip(img,flipCode=0)
# horizontal_vertical_flipping=cv2.flip(img,flipCode=-1)
# cv2.imshow('Horizontal Flipping',horizontal_flipping)
# cv2.imshow('Vertical Flipping',vertical_flipping)
# cv2.imshow('Horizontal-Vertical Flipping',horizontal_vertical_flipping)
#
# cv2.imshow('Horizontal Flipping',horizontal_flipping)
# cv2.imshow('Vertical Flipping',vertical_flipping)
# cv2.imshow('Horizontal-Vertical Flipping',horizontal_vertical_flipping)
# cv2.imshow('Original',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 08-border-constant:

'''If we want to add some border to an image (with constant height and width and color code) , we act like following code. we call this 
process "Padding", and in this video constant-padding is investigated.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\20\20.png')
#
# # Adding Border onto image's margin
# img=cv2.copyMakeBorder(img,top=50,bottom=50,left=50,right=50,borderType=cv2.BORDER_CONSTANT,value=(255,0,0)) # we padded same 50 pixel from
# # image's margin. and use Constant padding as border type. we also consider blue color as constant border.
# cv2.imshow('OpenCV Logo with cte Border',img)
# cv2.imwrite(r'C:\Users\rezaj\PycharmProjects\Koolac\OpenCV Logo with cte Border.png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# 09-border-replicate:

'''"Replicate" means repeating. we can expand our image's border by their last pixels color on the margins, instead of constant padding of them.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\20\20.png')
#
# # Adding Border onto image's margin
# img=cv2.copyMakeBorder(img,50,50,50,50,borderType=cv2.BORDER_REPLICATE) # We choose <<cv2.BORDER_REPLICATE>>
# cv2.imshow("OpenCV Logo Replicated",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# 10-border-reflect:

'''We can reflect the margins of our image when we expand the image border. <<cv2.BORDER_REFLECT>>, will start reflecting and padding from Last pixel,
and <<cv2.BORDER_REFLECT_101>> and <<cv2.BORDER_DEFAULT>> will start reflecting (padding) from one pixel to last.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\20\20.png')
#
# # Adding Border onto image's margin
# img=cv2.copyMakeBorder(img,50,50,50,50,borderType=cv2.BORDER_REFLECT) # We choose <<cv2.BORDER_REFLECT>>
# cv2.imshow("OpenCV Logo Reflected",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 11-border-wrap:

'''we can also clone the image (copy-paste the image) in around of margins ,when we are going to pad the borders. we use 
<<cv2.BORDER_WRAP>> ,seems the "wrap" word is not fit for this job prperly !! '''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\20\20.png')
#
# # Adding Border onto image's margin
# img=cv2.copyMakeBorder(img,50,50,50,50,borderType=cv2.BORDER_WRAP) # We choose <<cv2.BORDER_WRAP>>
# cv2.imshow("OpenCV Logo Wrapped",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 12-Split and Merge:

'''We can also split our image array to its BGR channels to find out what values of this three primary colors contain in each pixel.
 we can also do vice versa of this process , i.e. can take three channel B , G and R for a whole image NxM pixel and merge this channel together and
 make an image.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\05\05.png')
# print('Image Array Shape: ',img.shape)
# print('Image Array:\n\n ',img)
# # Splitting image to its primary colors
# B,G,R=cv2.split(img)
# print('Pixels (i.e.: corresponded index of row and column) contain Blue (w.r.t. Blue value):\n\n',B)
# print('Pixels (i.e.: corresponded index of row and column) contain Green (w.r.t. G value):\n\n',G)
# print('Pixels (i.e.: corresponded index of row and column) contain Red (w.r.t. Blue value):\n\n',R)
# cv2.imshow("4x4 BGR",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Making new image with B , G and R channels
# new_img=cv2.merge((B,G,R))
# cv2.imshow("New 4x4 BGR",new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 13-add:

'''In this video we learn how to add and combine two images together with "equal weight". in next video we discuss about non-equal weighted image.'''

# import cv2
#
# ant=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png') # Ant image
# text=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\02\02.png') # Text image
#
# # Adding two images:
# img=cv2.add(ant,text)
#
# cv2.imshow("Nimble Ant with Text",img)
# cv2.imwrite(r'C:\Users\rezaj\PycharmProjects\Koolac\Nimble Ant with Text.png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








# 14-addWeighted:

'''As we know every Image is a NumPy array, and we saw we can add and combine two image togheter with the same wight. we can add two array
be like: 

        alphaImg1 + bettaImg2 + gamma

where alpha is weight of Img1, betta is weight of Img2 and gamma is a offset. the output is a new numpy array and give us that image which discussed 
above.

Note that images MUST have same dimensions (height and width)'''


# import cv2
#
# ant=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png') # Ant image
# text=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\02\02.png') # Text image
#
# # Adding two images with different weight:
# img=cv2.addWeighted(src1=ant,alpha=0.9,src2=text,beta=0.8,gamma=60)
#
# cv2.imshow("Nimble Ant with Text with different weight",img)
# # cv2.imwrite(r'C:\Users\rezaj\PycharmProjects\Koolac\Nimble Ant with Text.png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()











'''04-Shapes and Text'''

# 01-line:

'''In this video we learn how to draw a line between two point with arbitrary coordinations.
Note that , here, we insert "Coordination" for introducing the points, NOT number of row and column. so we insert (x,y) as what we have in mathematics.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\03\03.png')
#
# # Drawing a line between point1 and point3
# img=cv2.line(img,pt1=(270,220),pt2=(170,270),color=(220,0,10),thickness=5)
#
# cv2.imshow('Line Drawing',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 02-arrowedLine:

'''If we want to draw an Arrowed Line or a vector instead of a line, we can do this following code.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\03\03.png')
#
# # Drawing a arrowing line from point3 to point1
#
# img=cv2.arrowedLine(img,pt1=(170,270),pt2=(270,220),color=(150,0,30),thickness=5)
# cv2.imshow('Arrowed Line Drawing',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 03-rectangle:

'''In this video , we learn how to draw an rectangle between two points. we can set the thickness value to -1 , to color up within the drawn rectangle.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\11\11.png')
#
# # Drawing a arrowing line from point3 to point1
#
# color=[150,0,250] #BGR System
# img=cv2.rectangle(img,pt1=(170,220),pt2=(270,320),color=color,thickness=5) # set thickness to -1 will color up inside the rectangle.
# cv2.imshow('Rectangle Drawing',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 04-circle:

'''In this video , we learn how to draw a circle with arbitrary central point and arbitrary radius. 
we can set the thickness value to -1 , to color up within the drawn circle.'''

# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\11\11.png')
#
# # Drawing a arrowing line from point3 to point1
#
# color=[150,0,250] #BGR System
# img=cv2.circle(img,center=(270,320),radius=100,color=color,thickness=5) # set thickness to -1 will color up inside the circle.
# cv2.imshow('Circle Drawing',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# 05-ellipse:

'''In this video , we learn how to draw a ellipse with arbitrary central point and arbitrary length of axes (Primary axis with the most length 
and secondary axis with the least length) and angle of rotation. we can set the thickness value to -1 , to color up within the drawn ellipse.'''


# import cv2
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\11\11.png')
#
# # Drawing a arrowing line from point3 to point1
#
# color=[150,0,250] #BGR System
# img=cv2.ellipse(img,center=(270,320),axes=(100,50),angle=30,startAngle=0,endAngle=300,color=color
#                 ,thickness=5) # set thickness to -1 will color up inside the ellipse. 100 pixel length of primary axis and 50 secondary.
# cv2.imshow('Ellipse Drawing',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# 06-putText:

'''In this video , we learn how to put a text on a image.'''

# import cv2
#
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\12\12.png')
#
# # Putting a text on coordination (170,270):
#
# font=cv2.FONT_HERSHEY_SIMPLEX # Considering a font for utilizing
# color=(95,4,230) # BGR System code
# img=cv2.putText(img=img,text='Koolac.Org',org=(170,270),fontFace=font,fontScale=1,color=color,thickness=2,lineType=cv2.LINE_AA) # org means origin of coordination
# cv2.imshow('Text put on img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









'''05-Trackbar'''



# 01-What is Trackbar:

'''The concept of trackbar has been explained.'''




# 02-Trackbar Example 01:

'''In this video , we learn how to make a simple trackbar. this trackbar window prints the input values as we change the bar to that corresponded values.'''

# import cv2
#
# def callback_func(x): # This function is going to use in cv2 trackbar and print the input value.
#     print(x)
#
#
# cv2.namedWindow('Koolac.org') # we first must define an empty window
# # Now we make a trackbar:
# cv2.createTrackbar('Value','Koolac.org',0,10,callback_func) # <value>: start point , <count>: final point.
# # note that never open and close parentheses for defined function here. Apperently ,if <value>'s value be less than 0, after adjusting you can't seek
# # for <value> below 0. Note that don't write <<name of argument = >> . :(  only insert parameteres otherwise it returns error.
# cv2.waitKey(0)
# cv2.destroyAllWindows()








# 03-Trackbar Example 02:

'''In this video we are going to put some text with trackbar on an IMAGE. it's a little bit different from last video. now let's code.'''


# import cv2
#
#
# def callback_func(x): # We have to build a useless function to insert in OpenCV <<creatTrackbar>> argument.
#     pass
#
#
# cv2.namedWindow('Koolac.Org') # make a window with "Koolac.Org" title.
# cv2.createTrackbar('Value','Koolac.Org',0,20,callback_func)
#
# stop=False  # Make a boolean class to control the <while> loop
#
# while stop==False:
#     img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\08\08.png')
#     value=cv2.getTrackbarPos('Value','Koolac.Org') # get current value of trackbar
#     img=cv2.putText(img=img,text=str(value),org=(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(230,115,30),thickness=2,lineType=cv2.LINE_AA)
#     cv2.imshow('Koolac.Org',img)
#     if cv2.waitKey(1)==ord('q'):
#         stop=True
# cv2.destroyAllWindows()






# 04-Trackbar Example 03:

'''In this video we learn how to change color system beside showing numbers above the image ,as we use trackbar.'''

# import cv2
#
# def callback_func(x):
#     pass
#
# cv2.namedWindow('Reza Javadzadeh')
# cv2.createTrackbar('Value','Reza Javadzadeh',0,10,callback_func)
# cv2.createTrackbar('System Color','Reza Javadzadeh',0,2,callback_func)
#
# stop=False
#
# while stop==False:
#     img=cv2.imread(r'C:\Users\rezaj\OneDrive\Desktop\Reza Javadzadeh.png')
#     img=cv2.putText(img,text=str(cv2.getTrackbarPos('Value','Reza Javadzadeh')),org=(50,50),fontFace=cv2.FONT_ITALIC,fontScale=2,
#                     color=(55,50,200),thickness=5)
#     if cv2.getTrackbarPos('System Color','Reza Javadzadeh')==0:
#        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        img = cv2.putText(img, text='B/W', org=(50, 150),
#                          fontFace=cv2.FONT_ITALIC, fontScale=2,color=(255, 50, 200), thickness=5)
#     elif cv2.getTrackbarPos('System Color','Reza Javadzadeh')==1:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.putText(img, text='RGB', org=(50, 150),
#                           fontFace=cv2.FONT_ITALIC, fontScale=2, color=(255, 50, 200), thickness=5)
#     else:
#         img = cv2.putText(img, text='BGR', org=(50, 150),
#                           fontFace=cv2.FONT_ITALIC, fontScale=2, color=(255, 50, 200), thickness=5)
#
#     cv2.imshow('Reza Javadzadeh',img)
#     if cv2.waitKey(1)==ord('q'):
#         stop=True
#
# cv2.destroyAllWindows()










'''06-Matplotlib and OpenCV'''







# 01-matplotlib-imread and imshow:

'''We can also read and show our image by using matplotlib. the shown image give us coordination too when we move around mouse cursor onto it where
cv2 didn't provide this option. look at following example.'''


# import cv2
# import matplotlib.pyplot as plt
#
#
#
# img=plt.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png') # reading the image
# plt.imshow(img) # Showing the image
# plt.show() # the Default for end of a matplotlib :)






# 02-matplotlib color system:

'''The difference between Matplotlib and OpenCV is that cv2 use BGR system channel which varies from 0 to 255 for each channel, and could be normalized
onto 0 to 1. But Matplotlib use RGB system color (or RGBA system color) which varies from 0 to 1 , and could be denormalized 0 to 255. the forth channel
is called "Alpha Channel" which is opposite of transparency. Alpha == 1 means that it looks like wall; you can't see behind of it and Alpha == 0 means 
that it looks like a glass; you can completely watch behind of it. '''





# 03-mixing OpenCV and Matplotlib:

'''We can exploit both OpenCV and Matplotlib concurrently , for instance , import our image by cv2 and showing it by matplotlib and vice versa.
But we should NOTE that we must change their system color correspond to module that we want to work with.'''


# import cv2
# import matplotlib.pyplot as plt
#
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png') # BGR is defaut system color for cv2
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Converting BGR to RGB format system color
# plt.imshow(img) # RGB is defaut system color for matplotlib
# plt.show()






# 04-grayscale image in matplotlib:

'''In this video we learn how to convert a colorful image onto a grayscale one by using matplotlib.'''


# import cv2
# import matplotlib.pyplot as plt
#
#
# img=plt.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # This time we convert fom RGB to gray-scaled
# plt.imshow(img,cmap='gray') # we MUST add "gray" as color map argument
# plt.show()
# # Some other color-map are used in matplotlib:
# # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'






# 05-image dtype under scrutiny:

'''We already know that dtype of cv2 arrays are <unit8> . but if we use matplotlib , the data type when we run .dtype is shown float32.'''






# 06-subplot:

'''We can show images in one window by using matplotlib and utilizing <<plt.subplot>>. the stateful method described in this video.'''

# import cv2
# import matplotlib.pyplot as plt
#
#
#
# img1=plt.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# img2=plt.imread(r'D:\Koolac\05- OpenCV\pictures\02\02.png')
#
# plt.subplot(1,2,1)
# plt.imshow(cv2.flip(cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY),1),cmap='gray')
#
# plt.subplot(1,2,2)
# plt.imshow(img2)
#
# plt.show()






# 07-stateless approach:

'''In this video stateless approach with matplotlib has been explained.'''

# import cv2
# import matplotlib.pyplot as plt
#
# img1=plt.imread(r'D:\Koolac\05- OpenCV\pictures\01\01.png')
# img2=plt.imread(r'D:\Koolac\05- OpenCV\pictures\02\02.png')
# img3=plt.imread(r'D:\Koolac\05- OpenCV\pictures\03\03.png')
# img4=plt.imread(r'D:\Koolac\05- OpenCV\pictures\04\04.png')
#
#
#
# fig,ax=plt.subplots(2,2)
#
# ax[0,0].imshow(img1)
# ax[0,1].imshow(img2)
# ax[1,0].imshow(img3)
# ax[1,1].imshow(img4)
#
# plt.show()





# 08-imread in depth:

'''We already know that , real gray-scaled image array doesn't have the third dimension which contains BGR channels. 
Or in images that have transparent background the third dimension have 4 channel, BGRA; where the "A" is corresponded to Alpha value.
1-When we are going to read an image with cv2 module, we can specify what kind of reading is under consideration;
2-Reading with actual shape of an array ; which can be gray scale, BGR , or BGRA. we determine it with code "-1".
3-Reading default; which the shape of an array disregarding to its real shape (i.e.: Gray-scaled ,BGR ,BGRA) would be read as BGR
and it has 3 channel. we can insert code "1" or insert nothing because this is a default mode.
Reading as a gray scaled image with code "0". 


'''

# import cv2
# import matplotlib.pyplot as plt
#
#
# img=cv2.imread(r'D:\Koolac\05- OpenCV\pictures\08\08-.png',-1)
# print(img.shape)
# cv2.imshow('koolac',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








'''07-Working with Webcam'''






# 01-frame meaning:

'''The frame concept and meaning has been explained.'''








# 02-VideoCapture:

'''In this video , we learn how to work with webcam to capture frame by OpenCV, and show it. '''


# import cv2
#
# webcam=cv2.VideoCapture(0) # we make an object which is named "webcam" or "cap". argument== 0 means the first webcam device connected on pc will be activated.
#
# while True:
#     ret,frame=webcam.read() # the <<.read>> method will read a frame and return it. "frame" variable is the frame array image, and "ret" will tell us
#     # the reading process conclude successfully or not with True or False value.
#     print(ret)
#     cv2.imshow('Koolac',frame) # Showing the frame.
#
#     if cv2.waitKey(1)==ord('q'):
#         break
#
# webcam.release() # will release (turn off from python enviroment) the webcam device.
# cv2.destroyAllWindows()







# 03-VideoCapture-Complementary:

'''In this video , we learn how to get some properties of a <VideoCapture> class. (like shape (height and width) of the class). and other
jobs we did to an image array we can do it same with a frame.'''


# import cv2
#
#
# cap=cv2.VideoCapture(0)# we make an object which is named "webcam" or "cap". argument== 0 means the first webcam device connected on pc will be activated.
# height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#
# print(f'Height: {height}\nWidth: {width}')
#
# while True:
#     ret,frame=cap.read()# the <<.read>> method will read a frame and return it. "frame" variable is the frame array image, and "ret" will tell us
# #   # the reading process conclude successfully or not with True or False value.
#     print(ret)
#     frame=cv2.putText(frame,'Reza Javadzadeh',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(255,0,200),2)
#     cv2.imshow('koolac',frame)
#     if cv2.waitKey(1)==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()





# 04-VideoCapture-from video file:

'''In this video we learn that to how read a video as a <<VideoCapture>> class and do the same things we have down to an image array to their 
frame as well and compute FPS (Frame per Second) index and its usefulness to compute proper argument for <<cv2.waitKey()>>. Calculating number of
all frame and then computing time of video running.'''



# import cv2
#
# cap=cv2.VideoCapture(r'D:\esta flash\Knock_Knock_2015_720p_(ValaMovie)_2.mkv')
#
# width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
# FPS=cap.get(cv2.CAP_PROP_FPS) #FPS
# N=cap.get(cv2.CAP_PROP_FRAME_COUNT) # Number of all frame
#
# print(f'Height: {height}\nWidth: {width}',end='\n\n')
# print(f'Runtime (m): {(N/FPS)/60}') # Runtime of video w.r.t. minutes (We should divide N to FPS)
#
# c=0
# while True:
#     ret,frame=cap.read()
#     frame=cv2.putText(frame,'Knock Knock (2015)-Keanu Reeves',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(255,10,200),2)
#     cv2.imshow('Movie',frame)
#     if c == 0:
#         print('Frames are show successfully')
#     c += 1
#     if cv2.waitKey(int(1000/FPS))==ord('q'): # We use 1000/FPS to calculate great time for delaying. NOTE that we MUST make it integer
#         break
# cap.release()
# cv2.destroyAllWindows()






# 05-Codec and FourCC:

'''In this video, "Codec" (Compressor and Decompressor) and "FourCC" (Four Character Code) has been explained. They have each 4 word for descibing.
 There are various type of FourCC, like "DivX" , "H264" which use MPEG-4 Codec algorithms. For more information about Codec which follow FourCC, 
 check www.fourcc.org '''





# 06-VideoWrite:

'''In this video , we learn how to write (save) a sequence of frames (video) with specific suffix ; like .mp4 ,.avi and etcetera. but we should 
first investigate whether which kind of codec with fourCC standard support that suffix? for better understanding look at the bellow code.'''


# import cv2
#
#
# cap=cv2.VideoCapture(r'D:\esta flash\Knock_Knock_2015_720p_(ValaMovie)_2.mkv')
# width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# FPS=cap.get(cv2.CAP_PROP_FPS) #FPS
# N=cap.get(cv2.CAP_PROP_FRAME_COUNT) # Number of all frame
#
#
# #out=cv2.VideoWriter(r'C:\Users\rezaj\PycharmProjects\Koolac\edited_video.mkv',-1,FPS,(640,480)) # first give the address to saving. we specify we want
# # .mkv again file as we typed in fill suffix.(we could choose .mp4, .avi, .m4v and..). second, we insert argument "-1" to return us what kind of fourCC
# # codec support this .mkv file. third, we give the fps that we actual seek and finally we add the desired dimension (resolution). Also we should note that
# # the address with those specific folder MUST exist, otherwise it wouldn't be saved.
# '''Now we run the program once to return what kind of fourCC's are suitable.'''
# # >>>
# # fourcc tag 0x34363248/'H264' codec_id 001B
# # fourcc tag 0x34363268/'h264' codec_id 001B
# # fourcc tag 0x34363258/'X264' codec_id 001B
# # fourcc tag 0x34363278/'x264' codec_id 001B
# # fourcc tag 0x31637661/'avc1' codec_id 001B
# # fourcc tag 0x43564144/'DAVC' codec_id 001B
# # fourcc tag 0x32564d53/'SMV2' codec_id 001B
# # fourcc tag 0x48535356/'VSSH' codec_id 001B
# # fourcc tag 0x34363251/'Q264' codec_id 001B
# # fourcc tag 0x34363256/'V264' codec_id 001B
# # fourcc tag 0x43564147/'GAVC' codec_id 001B
# # fourcc tag 0x56534d55/'UMSV' codec_id 001B
# # fourcc tag 0x64687374/'tshd' codec_id 001B
# # fourcc tag 0x434d4e49/'INMC' codec_id 001B
# # fourcc tag 0x33363248/'H263' codec_id 0004
# # fourcc tag 0x33363258/'X263' codec_id 0004
# # fourcc tag 0x33363254/'T263' codec_id 0004
# # fourcc tag 0x3336324c/'L263' codec_id 0004
# # fourcc tag 0x4b315856/'VX1K' codec_id 0004
# # fourcc tag 0x6f47795a/'ZyGo' codec_id 0004
# # fourcc tag 0x3336324d/'M263' codec_id 0004
# # fourcc tag 0x6d76736c/'lsvm' codec_id 0004
# # fourcc tag 0x33363248/'H263' codec_id 0013
# # fourcc tag 0x33363249/'I263' codec_id 0014
# # fourcc tag 0x31363248/'H261' codec_id 0003
# # fourcc tag 0x33363255/'U263' codec_id 0004
# # fourcc tag 0x344d5356/'VSM4' codec_id 0004
# # fourcc tag 0x34504d46/'FMP4' codec_id 000C
# # fourcc tag 0x58564944/'DIVX' codec_id 000C
# # fourcc tag 0x30355844/'DX50' codec_id 000C
# # fourcc tag 0x44495658/'XVID' codec_id 000C
# # fourcc tag 0x5334504d/'MP4S' codec_id 000C
# # fourcc tag 0x3253344d/'M4S2' codec_id 000C
# '''For example , we choose X264 codec fourCC. Now we have: '''
#
# fourCC=cv2.VideoWriter_fourcc('X','2','6','4') # X264.   we can add this argument like: *"X264"  too.
# out=cv2.VideoWriter(r'C:\Users\rezaj\PycharmProjects\Koolac\edited_video.mkv',fourCC,FPS,(640,480))
#
#
# c=0
# while True:
#     ret, frame = cap.read()
#     frame=cv2.putText(frame,'Knock Knock (2015)-Keanu Reeves',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(255,10,200),2)
#     if c == 0:
#         print('Frames are show successfully')
#     c += 1
#     if ret==True:
#         out.write(frame)
#     cv2.imshow('Movie', frame)
#     if cv2.waitKey(int(1000/FPS))==ord('q'): # We use 1000/FPS to calculate great time for delaying. NOTE that we MUST make it integer
#         break
# cap.release()
# cv2.destroyAllWindows()







