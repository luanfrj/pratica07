
# -----------------------------------------------------------------------------
# Prática 07
# Processamento De Imagens Aplicado a Visão Computacional - PGEE IFES 2021
# Aluno: Luan Ferreira Reis de Jesus
# -----------------------------------------------------------------------------

# importando as bibliotcas
import numpy as np
import cv2 as cv

debug = False

def rastreiaRetangulo(tck_win, hsv, h_min, h_max, s_min, v_min):
  mask = cv.inRange(hsv, np.array((h_min, s_min,v_min)), np.array((h_max,255.,255.)))
  roi_hist = cv.calcHist([hsv],[0],mask,[180],[0,180])
  cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

  dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

  if (debug):
    frame_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    frame_bgr = cv.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    cv.imshow('Track' + str(h_min), frame_bgr)

  #applying camshift to get the new location
  ret, tck_win = cv.CamShift(dst, tck_win, term_crit)
  return (ret, tck_win)

def calculaCentroBox(pts):
  sx = 0
  sy = 0
  for pt in pts:
    sx = sx + pt[0]
    sy = sy + pt[1]
  return(int(sx/4), int(sy/4))

def detectaQuadrante(pt, width, height):
  x = pt[0]
  y = pt[1]
  quadrante = 0

  if ((x > 0) and (x < width/2)) and ((y > 0) and (y < height/2)):
    quadrante = 1
  if ((x > width/2) and (x < width)) and ((y > 0) and (y < height/2)):
    quadrante = 2
  if ((x > 0) and (x < width/2)) and ((y > height/2) and (y < height)):
    quadrante = 3
  if ((x > width/2) and (x < width)) and ((y > height/2) and (y < height)):
    quadrante = 4
  return quadrante

def realcaTransicoes(frame):
  hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
  mask = np.matrix('-1 -1 -1; -1 8 -1,; -1 -1 -1')
  img_laplaciano = cv.filter2D(hsv[:,:,2].astype(np.float32),-1,mask)
  img_laplaciano = np.where(img_laplaciano < 0, 0, img_laplaciano)
  img_laplaciano = hsv[:,:,2].astype(np.float32) + img_laplaciano
  img_laplaciano = np.where(img_laplaciano > 255, 255, img_laplaciano)
  hsv[:,:,2] = img_laplaciano.astype(np.uint8)
  return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def suavizaImagem(frame):
  mask = np.matrix('1 1 1; 1 1 1,; 1 1 1') * (1/9)
  img_suavizada = cv.filter2D(frame.astype(np.float32),-1,mask)
  return img_suavizada

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_SETTINGS, 1)

__, frame = cap.read()

height, width, channels = frame.shape
tck_win1 = (0,0,height,width)
tck_win2 = (0,0,height,width)

# Cor verde
h_min1 = 105/2.
h_max1 = 120/2.
s_min1 = 100.
v_min1 = 100.

# Cor vermelha
h_min2 = 0/2.
h_max2 = 2/2.
s_min2 = 200.
v_min2 = 100.

#Setup the termination criteria, either 10 iteration
#or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(True):

  ret, frame = cap.read()

  hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
  hsv[:,:,0] = np.where(hsv[:,:,0] > 358/2, 0, hsv[:,:,0])

  # Faz o tracking do objeto verde
  ret1, tck_win1 = rastreiaRetangulo(tck_win1, hsv, h_min1, h_max1, s_min1, v_min1)
  
  pts = cv.boxPoints(ret1)
  pts = np.int0(pts)
  cX, cY = calculaCentroBox(pts)
  qverde = detectaQuadrante([cX, cY], width, height)
  cv.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
  if (debug):
    frame = cv.polylines(frame,[pts],True, 255,2)

  # Faz o tracking do objeto vermelho
  ret2, tck_win2 = rastreiaRetangulo(tck_win2, hsv, h_min2, h_max2, s_min2, v_min2)

  pts = cv.boxPoints(ret2)
  pts = np.int0(pts)
  cX, cY = calculaCentroBox(pts)
  qvermelho = detectaQuadrante([cX, cY], width, height)
  cv.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
  if (debug):
    frame = cv.polylines(frame,[pts],True, 255,2)

  if (qvermelho == 1 or qverde == 1):
    if (qvermelho == 1):
      frame[0:int(height/2), 0:int(width/2)] = realcaTransicoes(frame[0:int(height/2), 0:int(width/2)])
    if (qverde == 1):
      frame[0:int(height/2), 0:int(width/2)] = suavizaImagem(frame[0:int(height/2), 0:int(width/2)])
  else:
    # Tranparência verde no primeiro quadrante
    frame_q1 = frame[0:int(height/2), 0:int(width/2)].astype(np.float32) + [0, 127, 0]
    frame_q1 = np.where(frame_q1 > 255, 255, frame_q1)
    frame[0:int(height/2), 0:int(width/2)] = frame_q1.astype(np.uint8)

  if (qvermelho == 2 or qverde == 2):
    if (qvermelho == 2):
      frame[0:int(height/2), int(width/2):width] = realcaTransicoes(frame[0:int(height/2), int(width/2):width])
    if (qverde == 2):
      frame[0:int(height/2), int(width/2):width] = suavizaImagem(frame[0:int(height/2), int(width/2):width])
  else:
    # Binariza o segundo quadrante
    frame_bin = cv.cvtColor(frame[0:int(height/2), int(width/2):width], cv.COLOR_BGR2GRAY)
    ret, frame_bin = cv.threshold(frame_bin,127,255,cv.THRESH_BINARY)
    struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    frame_bin = cv.morphologyEx(frame_bin, cv.MORPH_CLOSE, struct_elem)
    frame_bin = np.stack((frame_bin, frame_bin, frame_bin), 2)
    frame[0:int(height/2), int(width/2):width] = frame_bin

  if (qvermelho == 3 or qverde == 3):
    if (qvermelho == 3):
      frame[int(height/2):height, 0:int(width/2)] = realcaTransicoes(frame[int(height/2):height, 0:int(width/2)])
    if (qvermelho == 3):
      frame[int(height/2):height, 0:int(width/2)] = suavizaImagem(frame[int(height/2):height, 0:int(width/2)])
  else:
    # Segmenta a pele humana no quadrante 3
    frame_hsv = cv.cvtColor(frame[int(height/2):height, 0:int(width/2)], cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, (10/2, 130, 50), (30/2, 255, 255))
    frame[int(height/2):height, 0:int(width/2)] = cv.bitwise_and(frame[int(height/2):height, 0:int(width/2)], frame[int(height/2):height, 0:int(width/2)], mask=mask)

  if (qvermelho == 4 or qverde == 4):
    if (qvermelho == 4):
      frame[int(height/2):height, int(width/2):width] = realcaTransicoes(frame[int(height/2):height, int(width/2):width])
    if (qvermelho == 4):
      frame[int(height/2):height, int(width/2):width] = suavizaImagem(frame[int(height/2):height, int(width/2):width])
  else:
    # Erode no quarto quadrante
    frame_gray = cv.cvtColor(frame[int(height/2):height, int(width/2):width], cv.COLOR_BGR2GRAY)
    struct_elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    frame_gray = cv.erode(frame_gray, struct_elem, iterations = 3)
    frame_gray = np.stack((frame_gray, frame_gray, frame_gray), 2)
    frame[int(height/2):height, int(width/2):width] = frame_gray

  # Desenha as linhas dividindo os quadrantes
  cv.line(frame, (0, int(height/2)), (width, int(height/2)), (0, 255, 0), thickness=2)
  cv.line(frame, (int(width/2), 0), (int(width/2), height), (0, 255, 0), thickness=2)
  cv.putText(frame, text='Luan 2021', org=(width - 100, height - 14), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1/2, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)


  cv.imshow('Resultado',frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
