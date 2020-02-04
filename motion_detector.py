import cv2,time, pandas
from datetime import datetime
first_frame=None #Assign Nothing to first frame. This will not give the error: variable not defined
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])
video=cv2.VideoCapture(0)

while True:

    check,frame = video.read()
    status=0 # 0 - No motion
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0) #Blurring the image to smoothen it, and remove noise

    if first_frame is None:
        first_frame = gray #first frame of video is supposed to be static, so that python can compare and identify moving objects later (1st Frame - Background)
        continue #goto next iteration

    delta_frame=cv2.absdiff(first_frame,gray) #Compare background frame (first frame) with current frame

    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1] #If the difference in motion at a pixel is greater than a threshold, make it white, else make it black

    #thresh_frame=cv2.dilate(thresh_frame, None, iterations=2) #Smoothen threshold frame, Mpre iterations - More Smoothness

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find and store contours (outline) of moving objects

    for contour in cnts:
        if cv2.contourArea(contour) < 5000: #If area of contour is less than 1000 search for the next contours
            continue
        status=1 #1 - Motion >2000 pixels present

        (x,y,w,h) = cv2.boundingRect(contour)
        print((x,y,w,h))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3) #Draw rectangle over contours greater than 1000 pixels

    status_list.append(status)
    status_list=status_list[-2:] #We only need last 2 statuses of the list.

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())

    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame", frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)
for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True) #Append start times and end times in data frame
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows()
