import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2 as cv
from utils_cv import load_images, show
import math


def save_char_to_png():
   b,g,r,a = 255,255,255,0
   characters = "QWERTYUIOPASDFGHJKLZXCVBNM1234567890"
   fontpath = "font/DIN_1451.ttf"     
   font = ImageFont.truetype(fontpath, 100)


   for c in characters:
      black = np.zeros((114,100,3),np.uint8)
      img_pil = Image.fromarray(black)
      draw = ImageDraw.Draw(img_pil)
      draw.text((10, 0),  str(c), font = font, fill = (b, g, r, a))
      img = np.array(img_pil)

   ## Display 
      cv.imshow("res", img)
      cv.waitKey(200)
      cv.destroyAllWindows()
      cv.imwrite("font/"+str(c)+".png", img)

def convert():
   characters = load_images("dane/font")

   for character in characters:
      org = cv.imread(character, cv.IMREAD_COLOR)
      img = cv.imread(character, cv.IMREAD_GRAYSCALE)
      contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      cv.fillPoly(img, [contours[0]], 255)
      x,y,w,h = cv.boundingRect(contours[0])
      # img = img[y:y+h, x:x+w]
      cv.drawContours(org, contours, 0, color=(0,0,255), thickness=3)
      cv.imwrite(character, img)

if __name__ == "__main__":
   # save_char_to_png()
   convert()

