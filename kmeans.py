# Code reference
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# https://stackoverflow.com/questions/55943596/check-only-particular-portion-of-video-feed-in-opencv
# https://stackoverflow.com/questions/39953263/get-video-dimension-in-python-opencv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    img = frame

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ROI = img[int(height*0.5):int(height*0.75), int(width*0.5):int(width*0.75)]
    cv2.imshow('ROI',ROI)
    cv2.imshow('frame',frame)
    
    ROI = ROI.reshape((ROI.shape[0] * ROI.shape[1],3))
    clt = KMeans(n_clusters=1) #cluster number
    clt.fit(ROI)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    

    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()