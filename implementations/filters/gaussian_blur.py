from PIL import Image, ImageFilter 

image = Image.open(r"FILE_PATH") 

# Cropping the image 
smol_image = image.crop((0, 0, 150, 150)) 

# Blurring on the cropped image 
blurred_image = smol_image.filter(ImageFilter.GaussianBlur) 

# Pasting the blurred image on the original image 
image.paste(blurred_image, (0,0)) 

# Displaying the image 
image.save('output.png') 

