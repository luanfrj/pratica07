
# -----------------------------------------------------------------------------
# Prática 07
# Processamento De Imagens Aplicado a Visão Computacional - PGEE IFES 2021
# Aluno: Luan Ferreira Reis de Jesus
# -----------------------------------------------------------------------------

# importando as bibliotcas
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_SETTINGS, 1)

__, frame = cap.read()

height, width, channels = frame.shape
tck_win = (0,0,height,width)

h_min = 205/2.
h_max = 220/2.
s_min = 100.
v_min = 50.

h_min2 = 340/2.
h_max2 = 355/2.
s_min2 = 130.
v_min2 = 50.

#Setup the termination criteria, either 10 iteration
#or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(True):
  #capturing frame-by-frame
  ret, frame = cap.read()

  #computing distribution tracked by camshift
  #camshift looks for the center of mass of 
  #this distribution
  hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

  # Faz o tracking do quadrado azul
  mask = cv.inRange(hsv, np.array((h_min, s_min,v_min)), np.array((h_max,255.,255.)))
  roi_hist = cv.calcHist([hsv],[0],mask,[180],[0,180])
  cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

  frame_blue = cv.bitwise_and(frame, frame, mask=mask)
  cv.imshow('Frame Blue',frame_blue)

  dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

  #applying camshift to get the new location
  ret, tck_win = cv.CamShift(dst, tck_win, term_crit)

  #drawing it on image
  pts = cv.boxPoints(ret)
  pts = np.int0(pts)
  #frame = cv.polylines(frame,[pts],True, 255,2)

  # Faz o tracking do quadrado vermelho
  mask = cv.inRange(hsv, np.array((h_min2, s_min2,v_min2)), np.array((h_max2,255.,255.)))
  roi_hist = cv.calcHist([hsv],[0],mask,[180],[0,180])
  cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

  frame_red = cv.bitwise_and(frame, frame, mask=mask)
  cv.imshow('Frame Red',frame_red)

  dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

  #applying camshift to get the new location
  ret, tck_win = cv.CamShift(dst, tck_win, term_crit)

  #drawing it on image
  pts = cv.boxPoints(ret)
  pts = np.int0(pts)
  #frame = cv.polylines(frame,[pts],True, 255,2)

  # Tranparência verde no primeiro quadrante
  frame[0:int(height/2), 0:int(width/2)] = frame[0:int(height/2), 0:int(width/2)] + [0, 20, 0]

  # Binariza o segundo quadrante
  frame_bin = cv.cvtColor(frame[0:int(height/2), int(width/2):width], cv.COLOR_BGR2GRAY)
  ret, frame_bin = cv.threshold(frame_bin,127,255,cv.THRESH_BINARY)
  frame_bin = np.stack((frame_bin, frame_bin, frame_bin), 2)
  frame[0:int(height/2), int(width/2):width] = frame_bin

  # Segmenta a pele humana no quadrante 3
  frame_hsv = cv.cvtColor(frame[int(height/2):height, 0:int(width/2)], cv.COLOR_BGR2HSV)
  mask = cv.inRange(frame_hsv, (10/2, 130, 50), (30/2, 255, 255))
  frame[int(height/2):height, 0:int(width/2)] = cv.bitwise_and(frame[int(height/2):height, 0:int(width/2)], frame[int(height/2):height, 0:int(width/2)], mask=mask)

  # Erode no quarto quadrante
  frame_gray = cv.cvtColor(frame[int(height/2):height, int(width/2):width], cv.COLOR_BGR2GRAY)
  struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
  frame_gray = cv.erode(frame_gray, struct_elem, iterations = 3)
  frame_gray = np.stack((frame_gray, frame_gray, frame_gray), 2)
  frame[int(height/2):height, int(width/2):width] = frame_gray

  # Desenha as linhas dividindo os quadrantes
  cv.line(frame, (0, int(height/2)), (width, int(height/2)), (0, 255, 0), thickness=2)
  cv.line(frame, (int(width/2), 0), (int(width/2), height), (0, 255, 0), thickness=2)


  cv.imshow('Resultado',frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
