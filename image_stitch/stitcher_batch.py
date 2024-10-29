import numpy as np
import cv2
import os
import imutils


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, imageA, imageB, ratio=0.85, reprojThresh=15.0, showMatches=False):
        
        # Print image shapes to debug
        print(f"Stitching images of shapes: {imageA.shape} and {imageB.shape}")

        # unpack the images, then detect keypoints and extract local invariant descriptors from them
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched keypoints to create a panorama
        if M is None:
            print("Not enough matches found between the images.")
            return None

        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M

        # get the dimensions of the input images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]


        # get the four corners of each image
        cornersA = np.float32([[0, 0], [0, hA], [wA, hA], [wA, 0]]).reshape(-1, 1, 2)
        cornersB = np.float32([[0, 0], [0, hB], [wB, hB], [wB, 0]]).reshape(-1, 1, 2)

        # transform the corners of imageA using the homography matrix H
        cornersA_trans = cv2.perspectiveTransform(cornersA, H)

        # find the minimum and maximum x and y coordinates
        all_corners = np.concatenate((cornersA_trans, cornersB), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # calculate the translation matrix to shift the image to the positive coordinates
        translation_dist = [-x_min, -y_min]
        translation_matrix = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # warp imageA with the homography matrix and the translation matrix
        result = cv2.warpPerspective(imageA, translation_matrix.dot(H), (x_max - x_min, y_max - y_min))

        # place imageB on the stitched output
        result[translation_dist[1]:hB + translation_dist[1], translation_dist[0]:wB + translation_dist[0]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # return a tuple of the stitched image and the visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # Print to debug
        print(f"Detecting and describing image of shape: {image.shape}")

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image using SIFT
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homography matrix and status of each matched point
            return (matches, H, status)

        # otherwise, no homography could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            print(f"Loaded image {img_path} of shape: {img.shape}")
            images.append(img)
        else:
            print(f"Failed to load image {img_path}")
            
    # # #rotate each image 90 degrees counterclockwise
    # images = [cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) for image in images]
    return images



def main():
    folder = 'wall_1'
    images = load_images_from_folder(folder)

    if len(images) == 0:
        print("No images found in the folder.")
        return

    stitcher = Stitcher()

    # Initialize the stitched image with the first image
    stitched = images[0]

    # Iterate over the remaining images and stitch them together
    for i in range(1, len(images)):
        print(f"Stitching image {i} with the previous result.")
        stitched_result = stitcher.stitch(stitched, images[i], showMatches=True)
        if stitched_result is None:
            print(f"Could not stitch image {i}.")
            return
        else:
            # Extract the stitched image if showMatches is True
            if isinstance(stitched_result, tuple):
                stitched = stitched_result[0]
                vis = stitched_result[1]
                # Show the matches visualization
                # cv2.imshow(f"Matches Visualization {i}", vis)
            else:
                stitched = stitched_result

    # # Show the final stitched image
    # cv2.imshow("Stitched Image", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the final stitched image
    cv2.imwrite('wall_1/stitch/stitched_result.jpg', stitched)


if __name__ == "__main__":
    main()
