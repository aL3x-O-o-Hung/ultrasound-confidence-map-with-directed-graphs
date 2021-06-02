"""
SRAD Filter + histogram matching + edge enhancement:
This filter adds histogram matching in each iteration of anisotropic diffusion, and adds a median filter
when the diffusion is ended
The diffusion coefficients are multiplied by a scaling constant to enhance the edge
The edge is detected by canny detector at higher level of laplacian pyramid. After the edge map is expanded to
original size, the point that is detected as edge have smaller scaling constant to slower the diffusion, and the point
that is not detected as edge have larger scaling constant to faster the diffusion.
"""

import numpy as np
import SimpleITK as sitk
import copy
from skimage.measure import compare_mse
from skimage.transform import pyramid_expand, pyramid_laplacian


class EdgeEnhancementHistogramMatchingSRADFilter:
    niter = None  # maximum number of iteration
    rect = []  # homogeneous ROI [x y width height]
    timestep = None  # update timestep
    errthre = 0  # when the mse between two iteration is smaller than the threshold, the iteration will be stopped
    edge_diffusion_scale = 1  # a constant to scale the diffusion coefficient at the edge point
    nonedge_diffusion_scale = 1  # a constant to scale the diffusion coefficient at the non-edge point

    def __init__(self, niter=None, rect=None, timestep=None, errthre=None, edge=None, nonedge=None):
        """
        :param niter: maximum number of iteration
        :param rect: [x, y, width, height], ROI of homogeneous tissue
        :param timestep: update time step
        :param errthre: threshold of MSE
        :param edge: edge diffusion scale
        :param nonedge: non-edge diffusion scale
        """
        if niter is not None:
            self.SetNumberOfIterations(niter)
        if rect is not None:
            self.SetROI(rect)
        if timestep is not None:
            self.SetTimeStep(timestep)
        if errthre is not None:
            self.SetMaximumError(errthre)
        if edge is not None and nonedge is not None:
            self.SetEdgeDiffusionScale(edge, nonedge)

    def SetNumberOfIterations(self, niter):
        """
        set maximum number of iteration
        :param niter: number of iteration
        """
        if niter > 0:
            self.niter = int(round(niter))
        else:
            raise ValueError('Number of iteration should be larger than 0')

    def SetMaximumError(self, err):
        """
        set maximum error in iteration, stop iteration when the error is below the threshold
        :param err: threshold that stops the iteration
        """
        if err > 0:
            self.errthre = err
        else:
            raise ValueError('Error threshold should be larger than 0')

    def SetTimeStep(self, timestep):
        """
        :param timestep: the time step in each iteration
        """
        if timestep > 0:
            self.timestep = timestep
        else:
            raise ValueError('Timestep should be larger then 0')

    def SetROI(self, rect):
        """
        set the homogeneous tissue region
        :param rect: [x, y, width, height]
        """
        if isinstance(rect, list) and len(rect) == 4:
            if all(x >= 0 for x in rect):
                self.rect = copy.deepcopy(rect)
            else:
                raise ValueError('ROI rectangle has element smaller than 0')
        elif type(rect) == np.ndarray and rect.size == 4:
            list_rect = list(rect.flatten)
            if all(x >= 0 for x in list_rect):
                self.rect = copy.deepcopy(list_rect)
            else:
                raise ValueError('ROI rectangle has element smaller than 0')
        else:
            raise TypeError('ROI rectangle has wrong type')

    def SetEdgeDiffusionScale(self, edge, nonedge):
        if edge > 0:
            self.edge_diffusion_scale = edge
        else:
            raise ValueError('Diffusion scale should be larger than 0')
        if nonedge > 0:
            self.nonedge_diffusion_scale = nonedge
        else:
            raise ValueError('diffusion scale should be larger than 0')


    def GetNumberOfIterations(self):
        return self.niter

    def GetROI(self):
        return self.rect

    def GetTimeStep(self):
        return self.timestep

    def Execute(self, image):
        """
        :param image: image to be filtered, simple itk image class
        :return: output: image after being filtered
        """
        if len(self.rect) == 0 or self.niter is None or self.timestep is None:
            raise ValueError('The filter parameters is not set appropriately')

        upper_x = self.rect[0]
        upper_y = self.rect[1]
        width = self.rect[2]
        height = self.rect[3]

        eps = 2e-16

        # set roi mask
        roi_seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        roi_seg.CopyInformation(image)
        for x in range(upper_x, upper_x + width + 1):
            for y in range(upper_y, upper_y + height + 1):
                roi_seg[x, y] = 1

        # min-max normalize image
        image = sitk.Cast(image, sitk.sitkFloat32)
        static_filter = sitk.MinimumMaximumImageFilter()
        static_filter.Execute(image)
        minimum = static_filter.GetMinimum()
        maximum = static_filter.GetMaximum()
        image = image - minimum
        image = image / (maximum - minimum)

        # image size
        M, N = image.GetSize()

        # log uncompress
        exp_filter = sitk.ExpImageFilter()
        exp_image = exp_filter.Execute(image)

        # grayscale histogram
        histogram_filter = sitk.HistogramMatchingImageFilter()
        histogram_filter.SetNumberOfHistogramLevels(8)
        histogram_filter.SetNumberOfMatchPoints(M * N // 10)
        origin_image = exp_image

        # generate edge mask and mask the edge location
        pyramid = tuple(pyramid_laplacian(sitk.GetArrayFromImage(exp_image), max_layer=3, downscale=2, multichannel=False))
        cannyfilter = sitk.CannyEdgeDetectionImageFilter()
        edge = cannyfilter.Execute(sitk.GetImageFromArray(pyramid[3]))
        edge = sitk.GetArrayFromImage(edge)
        edge = sitk.GetImageFromArray(pyramid_expand(edge, upscale=2 ** 3, multichannel=False))
        edge_mask = sitk.Image(image.GetSize(), sitk.sitkFloat64)
        edge_mask.CopyInformation(image)
        pyramidM, pyramidN = edge.GetSize()
        for x in range(pyramidM):
            for y in range(pyramidN):
                if x < M and y < N:
                    if edge[x, y] > 0.5:
                        edge_mask[x, y] = self.edge_diffusion_scale
                    else:
                        edge_mask[x, y] = self.nonedge_diffusion_scale

        # start iteration
        cur_iter = 0
        err = 10000000

        while cur_iter < self.niter and err > self.errthre:
            cur_iter += 1
            # speckle scale function
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(exp_image, roi_seg)
            mean_I = stats.GetMean(1)
            var_I = stats.GetVariance(1)
            q0_squared = var_I / (mean_I ** 2)

            # image difference
            exp_arr = sitk.GetArrayFromImage(exp_image)
            pre_arr = copy.deepcopy(exp_arr)

            W_array = np.hstack((np.reshape(exp_arr[:, 0], (N, 1)), exp_arr[:, 0:M - 1]))
            E_array = np.hstack((exp_arr[:, 1:M], np.reshape(exp_arr[:, M - 1], (N, 1))))
            N_array = np.vstack((np.reshape(exp_arr[0, :], (1, M)), exp_arr[0:N - 1, :]))
            S_array = np.vstack((exp_arr[1:N, :], np.reshape(exp_arr[N - 1, :], (1, M))))

            # calculate first order derivative
            IN = sitk.GetImageFromArray(N_array)
            IN.CopyInformation(exp_image)
            IS = sitk.GetImageFromArray(S_array)
            IS.CopyInformation(exp_image)
            IW = sitk.GetImageFromArray(W_array)
            IW.CopyInformation(exp_image)
            IE = sitk.GetImageFromArray(E_array)
            IE.CopyInformation(exp_image)
            dN = IN - exp_image
            dS = IS - exp_image
            dW = IW - exp_image
            dE = IE - exp_image

            # normalized discrete gradient magnitude
            G2 = (dN ** 2 + dS ** 2 + dW ** 2 + dE ** 2) / exp_image ** 2

            # normalized discrete laplacian
            L = (dN + dS + dW + dE) / exp_image

            # coefficient of variance
            num = (0.5 * G2) - ((1. / 16.) * (L ** 2))
            den = (1 + (0.25 * L)) ** 2
            q_squared = num / (den + eps)

            # diffusion coefficient
            den = (q_squared - q0_squared) / (q0_squared * (1 + q0_squared) + eps)
            c = 1 / (1 + den)
            # multiply edge preserving scaling factor
            c = c * edge_mask

            # saturate diffusion coefficient
            cM, cN = c.GetSize()
            for m in range(cM):
                for n in range(cN):
                    seed = (m, n)
                    if c[seed] > 1:
                        c[seed] = 1
                    elif c[seed] < 0:
                        c[seed] = 0

            # calculate update map
            c_array = sitk.GetArrayFromImage(c)
            cS_array = np.vstack((c_array[1:N, :], np.reshape(c_array[N - 1, :], (1, M))))
            cE_array = np.hstack((c_array[:, 1:], np.reshape(c_array[:, M - 1], (N, 1))))
            cS = sitk.GetImageFromArray(cS_array)
            cS.CopyInformation(c)
            cE = sitk.GetImageFromArray(cE_array)
            cE.CopyInformation(c)
            D = (c * dN) + (cS * dS) + (c * dW) + (cE * dE)

            # update
            exp_image += (self.timestep / 4.) * D

            # histogram correction
            exp_image = histogram_filter.Execute(exp_image, origin_image)
            # calculate error between iteration
            err = compare_mse(pre_arr, sitk.GetArrayFromImage(exp_image))


        # log compression
        log_filter = sitk.LogImageFilter()
        output = log_filter.Execute(exp_image)

        # median filter
        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(3)
        output = median_filter.Execute(output)

        return output
