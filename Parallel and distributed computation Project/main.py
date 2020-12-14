import cv2
import parallelCode
import serialCode
import time




# cap = cv2.VideoCapture

destPath = r"processedVideos/"

# מקבל את הנתיב ושם- פונקציית העיבוד
def DetectSaveVideo(cap, fileName):
    # קריאת הנתיב והדלקת הסרטון
    capture = cv2.VideoCapture(cap)
    # Get the Default resolutions- frame sizes,  check the frame width and height
    # גודל מסך הסרטון לצורך השמירה
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    # הגדרת סוג הסאטון שישמר - webm
    fourcc = cv2.VideoWriter_fourcc('V', 'P', '8', '0')

    # קביעת הסיומת לסרטון שהועלה
    li = list(fileName.split("."))
    fileName = li[0] + '.webm'
    # הגדרת השמירה- נתיב - תקיית הסרטונים המעובדים, סוג, גדלים
    out = cv2.VideoWriter(destPath + fileName, fourcc, 10,
                          (frame_width, frame_height))

    # Check if camera opened successfully
    if (capture.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (capture.isOpened()):
        # Capture frame-by-frame
        # capture.read- returns a bool (True/False). If frame is read correctly= ret,
        # מחזירה שני משתנים אחד בוליאני אם הצליח לקרא את הסרטון ובframe את מה שקרא
        ret, frame = capture.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # import the function that recognize smile and face
            canvas = serialCode.detect(gray, frame)

            # threads code
            # canvas = parallelCode.detect(gray, frame)
            # # Display the resulting frame
            cv2.imshow('Frame', canvas)

            # write the  frame
            out.write(canvas)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                return destPath + fileName
                break

        # Break the loop
        else:
            return destPath + fileName
            break



# print(datetime.now())
start = time.time()

print ("serial")
# print('parallel')
if __name__ == '__main__':
    DetectSaveVideo(
        r"C:\Users\henri\Pictures\Camera Roll\WIN_20200621_12_08_29_Pro_Trim.mp4", 'try.mp4')

# DetectSaveVideo(r"C:\Users\henri\Documents\Batia Zinger\מבני נתונים-אלגוריתמים\מבחן 2\WEEK 3\Videos\3-Binary Trees.mp4",'image3.mp4')
# DetectSaveVideo(r"C:\Users\henri\Documents\Batia Zinger\work interview_Trim.mp4", 'tryy.mp4')
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")


# print(datetime.now())
# DetectSaveVideo(0,'a.mp4')

cv2.destroyAllWindows()
