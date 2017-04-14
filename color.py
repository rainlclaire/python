// Process one frame
void segmentation::segmentBackground(Mat &image, list<percepUnit> &percepUnitsBackground) {

    Mat segments, labels;
    gpu::GpuMat gpuImageRGB, gpuImageHSV, gpuClose, gpuOpen;
    double t1, t2;

    t1 = getTime();

    // Do hard image processing on the GPU: (only works on micro!)
    gpuImageRGB.upload(image);
    gpu::cvtColor(gpuImageRGB, gpuImageHSV, CV_RGB2HSV, 4); // 1 or 4 channels for gpu operation

    // Morphology
    // TODO could use less variables to save texture memory.
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2) ); // was 5x5 at 2,2
    gpu::morphologyEx(gpuImageHSV, gpuClose, MORPH_CLOSE, element);
    gpu::morphologyEx(gpuClose, gpuOpen, MORPH_OPEN, element);

    // Mean shift
    TermCriteria iterations = TermCriteria(CV_TERMCRIT_ITER, 2, 0);
    gpu::meanShiftSegmentation(gpuOpen, segments, 10, 20, 300, iterations);

    // convert to greyscale
    vector<Mat> channels;
    split(segments, channels);

    // get labels from histogram of image.
    int size = 256;
    labels = Mat(256, 1, CV_32SC1);
    calcHist(&channels.at(2), 1, 0, Mat(), labels, 1, &size, 0);

    // Loop through hist
    int numROIs=0;
    for (int i=0; i<256; i++) {
        // If this bin matches a label.
        if (labels.at<float>(i) > 0) {
            // find areas of the image that match this label and findConours on the result.
            //stringstream ss;
            Mat label = Mat(channels.at(2).rows, channels.at(2).cols, CV_8UC1, Scalar::all(i)); // image filled with label colour.
            Mat boolImage = (channels.at(2) == label); // which pixels in labeled image are identical to this label?
            vector<vector<Point>> labelContours;
            findContours(boolImage, labelContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            // Loop through contours.
            for (int idx = 0; idx < labelContours.size(); idx++) {
                // get bounds for this contour.
                Rect bounds = boundingRect(labelContours[idx]);
                float area = contourArea(labelContours[idx]);

                // filter strange results (new segmentation may not cause any!)
                if (bounds.width < 1920 and bounds.height < 1080 and bounds.width > 10 and bounds.height > 10) {

                    // add padding to bounding box.
                    // TODO this count be a function! (extend scaleRect?)
                    if (bounds.x-imagePadding >= 0)
                        bounds.x -= imagePadding;
                    else
                        bounds.x = 0;
                    if (bounds.y-imagePadding >= 0)
                        bounds.y -= imagePadding;
                    else
                        bounds.y = 0;

                    int scalePadding = imagePadding*2;
                    int leftEdge = bounds.width+bounds.x;
                    int bottomEdge = bounds.height+bounds.y;
                    if (leftEdge+scalePadding <= 1920)
                        bounds.width += scalePadding;
                    else
                        bounds.width += 1920-leftEdge;
                    if (bottomEdge+scalePadding <= 1080)
                        bounds.height += scalePadding;
                    else
                        bounds.height += 1080-bottomEdge;

                    // create percepUnit
                    Mat patchROI = image(bounds);
                    Mat maskROI = boolImage(bounds);

                    percepUnit thisUnit = percepUnit(patchROI, maskROI, bounds.x, bounds.y, bounds.width, bounds.height, area);
                    thisUnit.calcColour(); // only do this on segmentation!
                    thisUnit.FGBG = "BG";
                    percepUnitsBackground.push_back(thisUnit); // Append percepUnit to background percept list.

                    numROIs++;      
                }
            }
        }
    }


    t2 = getTime();

    // TODO add a debug mode.
    cout << "Module Processing Time: backgroundSegmentation: " << t2-t1 << endl;
    cout << "Total Regions: " << numROIs << endl;

    // release GpuMats
    gpuImageRGB.release();
    gpuImageHSV.release();
    gpuClose.release();
    gpuOpen.release();
}