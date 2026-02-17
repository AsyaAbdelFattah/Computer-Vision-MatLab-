classdef Task1_AsyaAbdelFattah__mfile < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        YCbCrDropDown                   matlab.ui.control.DropDown
        YCbCrDropDownLabel              matlab.ui.control.Label
        LabDropDown                     matlab.ui.control.DropDown
        LabDropDownLabel                matlab.ui.control.Label
        HSIDropDown                     matlab.ui.control.DropDown
        HSIDropDownLabel                matlab.ui.control.Label
        RGBDropDown                     matlab.ui.control.DropDown
        RGBDropDownLabel                matlab.ui.control.Label
        IntensityBrightnessSlider       matlab.ui.control.Slider
        IntensityLabel                  matlab.ui.control.Label
        imagesharpeningDropDown_2       matlab.ui.control.DropDown
        imagesharpeningDropDown_2Label  matlab.ui.control.Label
        imagesmoothingDropDown_2        matlab.ui.control.DropDown
        imagesmoothingDropDown_2Label   matlab.ui.control.Label
        imagesharpeningDropDown         matlab.ui.control.DropDown
        imagesharpeningDropDownLabel    matlab.ui.control.Label
        imagesmoothingDropDown          matlab.ui.control.DropDown
        imagesmoothingDropDownLabel     matlab.ui.control.Label
        LoadImageButton                 matlab.ui.control.Button
        histogramequalizationCheckBox   matlab.ui.control.CheckBox
        ImageEnhancementLabel           matlab.ui.control.Label
        ResetButton                     matlab.ui.control.Button
        FrequencydomainfilteringLabel   matlab.ui.control.Label
        ColorSpaceconversionLabel       matlab.ui.control.Label
        SpatialdomainFilteringLabel     matlab.ui.control.Label
        UIAxes2                         matlab.ui.control.UIAxes
        UIAxes                          matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        OriginalImage
        ProcessedImage
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: ResetButton
        function ResetButtonPushed3(app, event)
            if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                return;
            end
            imshow(app.OriginalImage, 'Parent', app.UIAxes);
            app.ProcessedImage = app.OriginalImage;
        end

        % Callback function
        function ImageClicked(app, event)
            
        end

        % Callback function
        function ProcessButtonPushed2(app, event)
            
        end

        % Value changed function: histogramequalizationCheckBox
        function histogramequalizationCheckBoxValueChanged(app, event)
            if isempty(app.OriginalImage)
                return;
            end
            I= app.OriginalImage;
            if app.histogramequalizationCheckBox.Value
                if size(I,3)==1
                    eq = adapthisteq(I);
                    imshow(eq,'Parent',app.UIAxes);
                    title(app.UIAxes,'Histogram Equalized');
                else 
                    HSV = rgb2hsv(I);
                    V = HSV(:, :, 3);
                    V_eq = adapthisteq(V);
                    HSV_mod = HSV;
                    HSV_mod(:, :, 3) = V_eq;
                    RGB_eq = hsv2rgb(HSV_mod);
                    imshow(RGB_eq, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Histogram Equalized ');
                end 
            else
                imshow(app.OriginalImage,'Parent',app.UIAxes);
                title(app.UIAxes,'Original Image');
            end
            
        end

        % Value changed function: imagesharpeningDropDown
        function imagesharpeningDropDownValueChanged(app, event)

            if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.imagesharpeningDropDown.Value = 'Non';
                return;
            end
            
            I = app.OriginalImage;
            if size(I, 3) == 3 
                I = rgb2gray(I);
            end
            choice = app.imagesharpeningDropDown.Value;
            A = 3.5; % boost factor
            
            switch choice
                case 'Non'
                    imshow(I, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
            
                case 'Laplacian filter - 1st'
                    Lp_1 = [0 -1 0; -1 4 -1; 0 -1 0]; 
                    J_14 = imfilter(double(I), Lp_1);
                    J_14 = im2uint8(I + J_14);
                    imshow(J_14,[], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Laplacian filter - 1st');
            
                case 'Laplacian filter - 2nd'
                    Lp_2 = [-1 -1 -1; -1 8 -1; -1 -1 -1];
                    J_15 = imfilter(double(I), Lp_2);
                    J_15 = im2uint8(I + J_15);
                    imshow(J_15,[], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Laplacian filter - 2nd');
            
                case 'Boosted Laplacian filter - 1st'
                    Lp_1_bst = [0 -1 0; -1 4+A -1; 0 -1 0];
                    J_16 = imfilter(double(I), Lp_1_bst);
                    J_16 = im2uint8(I + J_16);
                    imshow(J_16,[], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Boosted Laplacian filter - 1st');
            
                case 'Boosted Laplacian filter - 2nd'
                    Lp_2_bst = [-1 -1 -1; -1 8+A -1; -1 -1 -1];
                    J_17 = imfilter(double(I), Lp_2_bst);
                    J_17 = im2uint8(I + J_17);
                    imshow(J_17, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Boosted Laplacian filter - 2nd');
            
                case 'Horizontal Sobel filter'
                    Sb_x = [-1 -2 -1; 0 0 0; 1 2 1];
                    J_18 = imfilter(double(I), Sb_x);
                    J_18 = im2uint8(I + J_18);
                    imshow(J_18, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Horizontal Sobel filter');
            
                case 'Vertical Sobel filter'
                    Sb_y = [-1 0 1; -2 0 2; -1 0 1];
                    J_19 = imfilter(double(I), Sb_y);
                    J_19 = im2uint8(I + J_19);
                    imshow(J_19, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Vertical Sobel filter');
            
                case 'Horizontal Prewitt filter'
                    Pw_x = [-1 -1 -1; 0 0 0; 1 1 1];
                    J_20 = imfilter(double(I), Pw_x);
                    J_20 = im2uint8(I + J_20);
                    imshow(J_20, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Horizontal Prewitt filter');
            
                case 'Vertical Prewitt filter'
                    Pw_y = [-1 0 1; -1 0 1; -1 0 1];
                    J_21 = imfilter(double(I), Pw_y);
                    J_21 = im2uint8(I + J_21);
                    imshow(J_21, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Vertical Prewitt filter');
            end            
        end

        % Value changed function: imagesharpeningDropDown_2
        function imagesharpeningDropDown_2ValueChanged(app, event)
            if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.imagesharpeningDropDown_2.Value = 'Non';
                return;
            end
        
            I = app.OriginalImage;
            if size(I, 3) == 3
                I = rgb2gray(I);
            end
            choice = app.imagesharpeningDropDown_2.Value;
            % Fourier Transform
            FT = fftshift(fft2(I));
            [m, n] = size(I);
            
            % Create frequency grid
            [X, Y] = meshgrid(1:n, 1:m);
            X = X - (n + 1)/2;
            Y = Y - (m + 1)/2;
            
            % Distance matrix
            D = sqrt(X.^2 + Y.^2);
            switch choice
                case 'Non '
                    imshow(I, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
            
                case 'Ideal'
                    IHPF=D > 60; 
                    J = ifft2(ifftshift(FT .* IHPF));
                    imshow(abs(J), [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Ideal Filter');
            
                case 'Gaussian high-pass filtering'
                    D0 = 30;
                    GHPF = 1-exp(-D.^2 / (2 * D0^2));
                    J = ifft2(ifftshift(FT .* GHPF));
                    imshow(abs(J), [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Gaussian HPF');
            
                case 'Butterworth'
                    cutoff = 60;
                    n = 2; 
                    BHPF = 1 ./ (1 + (cutoff./ D).^(2*n));
                    J = ifft2(ifftshift(FT .* BHPF));
                    imshow(abs(J), [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Butterworth HPF');
            end
        end

        % Value changed function: imagesmoothingDropDown
        function imagesmoothingDropDownValueChanged(app, event)
           if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.imagesmoothingDropDown.Value = 'Non';
                return;
            end
                
            I = app.OriginalImage;
            choice = app.imagesmoothingDropDown.Value;
        
            switch choice
                case 'Non'
                    imshow(I, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
        
                case 'Box filter - size 3'
                    H_box_3 = (1/9).*[1 1 1; 1 1 1; 1 1 1]; % box filter with size 3x3
                    J_2 = imfilter(double(I),H_box_3);
                    imshow(J_2, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Box filter - size 3');
                
        
                case 'Box filter - size 5'
                    H_box_5 = (1/25).* ones(5,5);
                    J_3 = imfilter(double(I),H_box_5);
                    imshow(J_3, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Box filter - size 5');

                case 'Box filter - size 7'
                    H_box_7 = (1/49).* ones(7,7);
                    J_4 = imfilter(double(I),H_box_7);
                    imshow(J_4, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Box filter - size 7');
                
                case 'Box filter - size 9'
                    H_box_9 = (1/81).* ones(9,9);
                    J_5 = imfilter(double(I),H_box_9);
                    imshow(J_5, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Box filter - size 9');
                
                case 'Average filter - size 3'
                    H_avg_3 = (1/16).*[1 2 1;
                                       2 4 2; 
                                       1 2 1]; 
                    J_6 = imfilter(double(I),H_avg_3);
                    imshow(J_6, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Average filter - size 3');
        
                case 'Average filter - size 5'  
                    H_avg_5 = (1/65).*[1 2 2 2 1; 
                                       1 2 4 2 1;
                                       2 4 8 4 2;
                                       1 2 4 2 1;
                                       1 2 2 2 1]; % average filter with size 5x5
                    J_7 = imfilter(double(I),H_avg_5);
                    imshow(J_7, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Average filter - size 5');

                
                case 'Average filter - size 7'  
                    H_avg_7 = (1/128).*[1 1 1 2 1 1 1; 
                                        1 1 2 4 2 1 1;
                                        1 2 4 8 4 2 1;
                                        2 4 8 16 8 4 2;
                                        1 2 4 8 4 2 1;
                                        1 1 2 4 2 1 1;
                                        1 1 1 2 1 1 1]; 
                    J_8 = imfilter(double(I),H_avg_7);
                    imshow(J_8 ,[], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Average filter - size 7');
               
                case 'Average filter - size 9'  
                    H_avg_9 = (1/280).*[1 1 1 1 2 1 1 1 1;
                                        1 1 1 2 4 2 1 1 1; 
                                        1 1 2 4 8 4 2 1 1;
                                        1 2 4 8 16 8 4 2 1;
                                        2 4 8 16 32 16 8 4 2;
                                        1 2 4 8 16 8 4 2 1;
                                        1 1 2 4 8 4 2 1 1;
                                        1 1 1 2 4 2 1 1 1;
                                        1 1 1 1 2 1 1 1 1]; % average filter with size 5x5
                    J_9 = imfilter(double(I),H_avg_9);
                    imshow(J_9, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Average filter - size 9');
                
                case 'Median filter - size 3'
                    J_10 = medfilt3(double(I), [3 3 1]);
                    imshow(J_10, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Median filter - size 3');

                
                case 'Median filter - size 5'
                    J_11 = medfilt3(double(I), [5 5 1]);
                    imshow(J_11, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Median filter - size 5');

                
                case 'Median filter - size 7'
                    J_12 = medfilt3(double(I), [7 7 1]);
                    imshow(J_12, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Median filter - size 7');

                
                case 'Median filter - size 9'
                    J_13 = medfilt3(double(I), [9 9 1]);
                    imshow(J_13, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Median filter - size 9');

            end 
        end

        % Value changed function: imagesmoothingDropDown_2
        function imagesmoothingDropDown_2ValueChanged(app, event)
            if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.imagesmoothingDropDown_2.Value = 'Non';
                return;
            end
        
            I = double(app.OriginalImage);
            if size(I, 3) == 3
                I = rgb2gray(I);
            end
            choice = app.imagesmoothingDropDown_2.Value;
            % Fourier Transform
            FT = fftshift(fft2(I));
            [m, n] = size(I);
            
            % Create frequency grid
            [X, Y] = meshgrid(1:n, 1:m);
            X = X - (n + 1)/2;
            Y = Y - (m + 1)/2;
            
            % Distance matrix
            D = sqrt(X.^2 + Y.^2);
            switch choice
                case 'Non '
                    imshow(I, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
            
                case 'Ideal'
                    ILPF =D < 60;
                    J = ifft2(ifftshift(FT .* ILPF));
                    imshow(abs(J), [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Ideal Filter');
            
                case 'Gaussian low-pass filtering'
                    D0 = 30;
                    GLPF = exp(-D.^2 / (2 * D0^2));
                    J = ifft2(ifftshift(FT .* GLPF));
                    imshow(abs(J), [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Gaussian LPF');
            
                case 'Butterworth'
                    cutoff = 60;
                    n = 1; 
                    BLPF = 1 ./ (1 + (D ./ cutoff).^(2*n));
                    J = ifft2(ifftshift(FT .* BLPF));
                    imshow(abs(J), [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Butterworth LPF');
            end

            
        end

        % Value changed function: IntensityBrightnessSlider
        function IntensityBrightnessSliderValueChanged(app, event)
            if isempty(app.OriginalImage)
                    return;
                end
                value = app.IntensityBrightnessSlider.Value;
                img = im2double(app.OriginalImage);
                bright = img + value/200;  
                bright = max(min(bright,1),0); 
                imshow(bright,'Parent',app.UIAxes);
                title(app.UIAxes,'Brightness Adjusted');
        end

        % Button pushed function: LoadImageButton
        function LoadImageButtonPushed(app, event)
            [file,path] = uigetfile({'*.jpg;*.png;*.tif'});
            if isequal(file,0)
                return;
            end
            app.OriginalImage = im2double(imread(fullfile(path,file)));
            imshow(app.OriginalImage, 'Parent', app.UIAxes2);
        end

        % Button down function: UIAxes
        function UIAxesButtonDown(app, event)
            
        end

        % Value changed function: HSIDropDown
        function HSIDropDownValueChanged(app, event)
             if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.HSIDropDown.Value = 'Non';
                return;
            end
                
            J = app.OriginalImage;
            
            if size(J, 3) == 1
                uialert(app.UIFigure, 'The image is grayscale, cannot convert to HSI.', 'Error');
                app.HSIDropDown.Value = 'Non';
                imshow(J, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Grayscale Image');
                return;
            end
            
            HSI=rgb2hsv(J);
            choice = app.HSIDropDown.Value;
        
            switch choice
                case 'Non'
                    imshow(J, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
                case 'Full'
                    imshow(HSI, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'HSI Image');
        
                case 'H'
                    H = HSI(:,:,1);
                    imshow(H, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Hue Channel');

                case 'S'
                    S = HSI(:,:,2);
                    imshow(S, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Saturation Channel');

                case 'I'
                    I = HSI(:,:,3);
                    imshow(I, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Intensity Channel');
            end 
        end

        % Value changed function: RGBDropDown
        function RGBDropDownValueChanged(app, event)
             if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.RGBDropDown.Value = 'Non';
                return;
            end
                
            J = app.OriginalImage;
       
            if size(J, 3) == 1
                uialert(app.UIFigure, 'The image is grayscale, cannot convert to HSI.', 'Error');
                app.HSIDropDown.Value = 'Non';
                imshow(J, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Grayscale Image');
                return;
            end

            choice = app.RGBDropDown.Value;
        
            switch choice
                case 'Non'
                    imshow(J, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
                case 'Full'
                    imshow(J, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'RGB Image');
        
                case 'R'
                    R = J(:,:,1);
                    imshow(R, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Red Channel');

                case 'G'
                    G = J(:,:,2);
                    imshow(G, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Green Channel');

                case 'B'
                    B = J(:,:,3);
                    imshow(B, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Blue Channel');
            end 
        end

        % Value changed function: YCbCrDropDown
        function YCbCrDropDownValueChanged(app, event)
            if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.YCbCrDropDown.Value = 'Non';
                return;
            end
            
            J = im2double(app.OriginalImage);


            if size(J, 3) == 1
                uialert(app.UIFigure, 'The image is grayscale, cannot convert to HSI.', 'Error');
                app.HSIDropDown.Value = 'Non';
                imshow(J, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Grayscale Image');
                return;
            end

            YCbCr = rgb2ycbcr(J);
            choice = app.YCbCrDropDown.Value;
            
            switch choice
                case 'Non'
                    imshow(J, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
                case 'Full'
                    imshow(YCbCr, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'YCbCr Image');
            
                case 'Y'
                    Y = YCbCr(:,:,1);
                    imshow(Y, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Luminance (Y)');
            
                case 'Cb'
                    Cb = YCbCr(:,:,2);
                    imshow(Cb, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Blue-difference (Cb)');
            
                case 'Cr'
                    Cr = YCbCr(:,:,3);
                    imshow(Cr, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Red-difference (Cr)');
            end

        end

        % Value changed function: LabDropDown
        function LabDropDownValueChanged(app, event)
            if isempty(app.OriginalImage)
                uialert(app.UIFigure,'Please load an image first!','Error');
                app.LabDropDown.Value = 'Non';
                return;
            end
            
            J = im2double(app.OriginalImage);

       
            if size(J, 3) == 1
                uialert(app.UIFigure, 'The image is grayscale, cannot convert to HSI.', 'Error');
                app.HSIDropDown.Value = 'Non';
                imshow(J, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Grayscale Image');
                return;
            end

            Lab = rgb2lab(J);
            choice = app.LabDropDown.Value;
            
            switch choice
                case 'Non'
                    imshow(J, 'Parent', app.UIAxes);
                    title(app.UIAxes, 'Original Image');
                case 'Full'
                    imshow(Lab, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'L*a*b* Image');
            
                case 'L'
                    L = Lab(:,:,1);
                    imshow(L, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'L Channel');
            
                case 'A'
                    A = Lab(:,:,2);
                    imshow(A, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'A Channel');
            
                case 'B'
                    B = Lab(:,:,3);
                    imshow(B, [], 'Parent', app.UIAxes);
                    title(app.UIAxes, 'B Channel');
            end

            
        end

        % Button down function: UIAxes2
        function UIAxes2ButtonDown(app, event)
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.9412 0.9412 0.9412];
            app.UIFigure.Position = [100 100 720 1013];
            app.UIFigure.Name = 'MATLAB App';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'Processed')
            app.UIAxes.XTick = [];
            app.UIAxes.XTickLabel = '';
            app.UIAxes.YTick = [];
            app.UIAxes.YTickLabel = '';
            app.UIAxes.ZTick = [];
            app.UIAxes.ButtonDownFcn = createCallbackFcn(app, @UIAxesButtonDown, true);
            app.UIAxes.Position = [370 690 314 296];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.UIFigure);
            title(app.UIAxes2, 'Original')
            app.UIAxes2.XTick = [];
            app.UIAxes2.XTickLabel = '';
            app.UIAxes2.YTick = [];
            app.UIAxes2.ZTick = [];
            app.UIAxes2.ButtonDownFcn = createCallbackFcn(app, @UIAxes2ButtonDown, true);
            app.UIAxes2.Position = [15 689 307 297];

            % Create SpatialdomainFilteringLabel
            app.SpatialdomainFilteringLabel = uilabel(app.UIFigure);
            app.SpatialdomainFilteringLabel.FontName = 'Arial';
            app.SpatialdomainFilteringLabel.FontSize = 16;
            app.SpatialdomainFilteringLabel.FontWeight = 'bold';
            app.SpatialdomainFilteringLabel.Position = [30 436 198 22];
            app.SpatialdomainFilteringLabel.Text = ' Spatial-domain Filtering:';

            % Create ColorSpaceconversionLabel
            app.ColorSpaceconversionLabel = uilabel(app.UIFigure);
            app.ColorSpaceconversionLabel.FontName = 'Arial';
            app.ColorSpaceconversionLabel.FontSize = 16;
            app.ColorSpaceconversionLabel.FontWeight = 'bold';
            app.ColorSpaceconversionLabel.Position = [30 224 199 22];
            app.ColorSpaceconversionLabel.Text = ' Color Space conversion:';

            % Create FrequencydomainfilteringLabel
            app.FrequencydomainfilteringLabel = uilabel(app.UIFigure);
            app.FrequencydomainfilteringLabel.FontName = 'Arial';
            app.FrequencydomainfilteringLabel.FontSize = 16;
            app.FrequencydomainfilteringLabel.FontWeight = 'bold';
            app.FrequencydomainfilteringLabel.Position = [30 326 222 22];
            app.FrequencydomainfilteringLabel.Text = ' Frequency-domain filtering:';

            % Create ResetButton
            app.ResetButton = uibutton(app.UIFigure, 'push');
            app.ResetButton.ButtonPushedFcn = createCallbackFcn(app, @ResetButtonPushed3, true);
            app.ResetButton.FontName = 'Arial';
            app.ResetButton.FontSize = 16;
            app.ResetButton.FontWeight = 'bold';
            app.ResetButton.Position = [399 649 257 30];
            app.ResetButton.Text = 'Reset';

            % Create ImageEnhancementLabel
            app.ImageEnhancementLabel = uilabel(app.UIFigure);
            app.ImageEnhancementLabel.FontName = 'Arial';
            app.ImageEnhancementLabel.FontSize = 16;
            app.ImageEnhancementLabel.FontWeight = 'bold';
            app.ImageEnhancementLabel.Position = [34 597 170 22];
            app.ImageEnhancementLabel.Text = ' Image Enhancement:';

            % Create histogramequalizationCheckBox
            app.histogramequalizationCheckBox = uicheckbox(app.UIFigure);
            app.histogramequalizationCheckBox.ValueChangedFcn = createCallbackFcn(app, @histogramequalizationCheckBoxValueChanged, true);
            app.histogramequalizationCheckBox.Text = 'histogram equalization';
            app.histogramequalizationCheckBox.FontName = 'Arial';
            app.histogramequalizationCheckBox.FontSize = 14;
            app.histogramequalizationCheckBox.FontWeight = 'bold';
            app.histogramequalizationCheckBox.Position = [57 547 174 22];

            % Create LoadImageButton
            app.LoadImageButton = uibutton(app.UIFigure, 'push');
            app.LoadImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadImageButtonPushed, true);
            app.LoadImageButton.FontName = 'Arial';
            app.LoadImageButton.FontSize = 14;
            app.LoadImageButton.FontWeight = 'bold';
            app.LoadImageButton.Position = [45 649 257 30];
            app.LoadImageButton.Text = 'Load Image';

            % Create imagesmoothingDropDownLabel
            app.imagesmoothingDropDownLabel = uilabel(app.UIFigure);
            app.imagesmoothingDropDownLabel.HorizontalAlignment = 'right';
            app.imagesmoothingDropDownLabel.FontName = 'Arial';
            app.imagesmoothingDropDownLabel.FontSize = 14;
            app.imagesmoothingDropDownLabel.FontWeight = 'bold';
            app.imagesmoothingDropDownLabel.Position = [31 396 125 22];
            app.imagesmoothingDropDownLabel.Text = ' image smoothing';

            % Create imagesmoothingDropDown
            app.imagesmoothingDropDown = uidropdown(app.UIFigure);
            app.imagesmoothingDropDown.Items = {'Non', 'Box filter - size 3', 'Box filter - size 5', 'Box filter - size 7', 'Box filter - size 9', 'Average filter - size 3', 'Average filter - size 5', 'Average filter - size 7', 'Average filter - size 9', 'Median filter - size 3', 'Median filter - size 5', 'Median filter - size 7', 'Median filter - size 9'};
            app.imagesmoothingDropDown.ValueChangedFcn = createCallbackFcn(app, @imagesmoothingDropDownValueChanged, true);
            app.imagesmoothingDropDown.FontName = 'Arial';
            app.imagesmoothingDropDown.FontSize = 14;
            app.imagesmoothingDropDown.Position = [171 396 145 22];
            app.imagesmoothingDropDown.Value = 'Non';

            % Create imagesharpeningDropDownLabel
            app.imagesharpeningDropDownLabel = uilabel(app.UIFigure);
            app.imagesharpeningDropDownLabel.HorizontalAlignment = 'right';
            app.imagesharpeningDropDownLabel.FontName = 'Arial';
            app.imagesharpeningDropDownLabel.FontSize = 14;
            app.imagesharpeningDropDownLabel.FontWeight = 'bold';
            app.imagesharpeningDropDownLabel.Position = [321 396 129 22];
            app.imagesharpeningDropDownLabel.Text = ' image sharpening';

            % Create imagesharpeningDropDown
            app.imagesharpeningDropDown = uidropdown(app.UIFigure);
            app.imagesharpeningDropDown.Items = {'Non', 'Laplacian filter - 1st', 'Laplacian filter - 2nd', 'Boosted Laplacian filter - 1st', 'Boosted Laplacian filter - 2nd', 'Horizontal Sobel filter', 'Vertical Sobel filter', 'Horizontal Prewitt filter', 'Vertical Prewitt filter'};
            app.imagesharpeningDropDown.ValueChangedFcn = createCallbackFcn(app, @imagesharpeningDropDownValueChanged, true);
            app.imagesharpeningDropDown.FontName = 'Arial';
            app.imagesharpeningDropDown.FontSize = 14;
            app.imagesharpeningDropDown.Position = [465 396 142 22];
            app.imagesharpeningDropDown.Value = 'Non';

            % Create imagesmoothingDropDown_2Label
            app.imagesmoothingDropDown_2Label = uilabel(app.UIFigure);
            app.imagesmoothingDropDown_2Label.HorizontalAlignment = 'right';
            app.imagesmoothingDropDown_2Label.FontName = 'Arial';
            app.imagesmoothingDropDown_2Label.FontSize = 14;
            app.imagesmoothingDropDown_2Label.FontWeight = 'bold';
            app.imagesmoothingDropDown_2Label.Position = [31 291 125 22];
            app.imagesmoothingDropDown_2Label.Text = ' image smoothing';

            % Create imagesmoothingDropDown_2
            app.imagesmoothingDropDown_2 = uidropdown(app.UIFigure);
            app.imagesmoothingDropDown_2.Items = {'Non', 'Ideal', 'Butterworth', 'Gaussian low-pass filtering'};
            app.imagesmoothingDropDown_2.ValueChangedFcn = createCallbackFcn(app, @imagesmoothingDropDown_2ValueChanged, true);
            app.imagesmoothingDropDown_2.FontName = 'Arial';
            app.imagesmoothingDropDown_2.FontSize = 14;
            app.imagesmoothingDropDown_2.Position = [171 291 145 22];
            app.imagesmoothingDropDown_2.Value = 'Non';

            % Create imagesharpeningDropDown_2Label
            app.imagesharpeningDropDown_2Label = uilabel(app.UIFigure);
            app.imagesharpeningDropDown_2Label.HorizontalAlignment = 'right';
            app.imagesharpeningDropDown_2Label.FontName = 'Arial';
            app.imagesharpeningDropDown_2Label.FontSize = 14;
            app.imagesharpeningDropDown_2Label.FontWeight = 'bold';
            app.imagesharpeningDropDown_2Label.Position = [320 291 129 22];
            app.imagesharpeningDropDown_2Label.Text = ' image sharpening';

            % Create imagesharpeningDropDown_2
            app.imagesharpeningDropDown_2 = uidropdown(app.UIFigure);
            app.imagesharpeningDropDown_2.Items = {'Non', 'Ideal', 'Butterworth', 'Gaussian high-pass filtering'};
            app.imagesharpeningDropDown_2.ValueChangedFcn = createCallbackFcn(app, @imagesharpeningDropDown_2ValueChanged, true);
            app.imagesharpeningDropDown_2.FontName = 'Arial';
            app.imagesharpeningDropDown_2.FontSize = 14;
            app.imagesharpeningDropDown_2.Position = [464 291 141 22];
            app.imagesharpeningDropDown_2.Value = 'Non';

            % Create IntensityLabel
            app.IntensityLabel = uilabel(app.UIFigure);
            app.IntensityLabel.HorizontalAlignment = 'right';
            app.IntensityLabel.FontName = 'Arial';
            app.IntensityLabel.FontSize = 14;
            app.IntensityLabel.FontWeight = 'bold';
            app.IntensityLabel.Position = [50 500 140 22];
            app.IntensityLabel.Text = 'Intensity Brightness';

            % Create IntensityBrightnessSlider
            app.IntensityBrightnessSlider = uislider(app.UIFigure);
            app.IntensityBrightnessSlider.MajorTicks = [0 10 20 30 40 50 60 70 80 90 100];
            app.IntensityBrightnessSlider.ValueChangedFcn = createCallbackFcn(app, @IntensityBrightnessSliderValueChanged, true);
            app.IntensityBrightnessSlider.Position = [210 519 385 3];

            % Create RGBDropDownLabel
            app.RGBDropDownLabel = uilabel(app.UIFigure);
            app.RGBDropDownLabel.HorizontalAlignment = 'right';
            app.RGBDropDownLabel.FontName = 'Arial';
            app.RGBDropDownLabel.FontSize = 14;
            app.RGBDropDownLabel.FontWeight = 'bold';
            app.RGBDropDownLabel.Position = [60 169 36 22];
            app.RGBDropDownLabel.Text = 'RGB';

            % Create RGBDropDown
            app.RGBDropDown = uidropdown(app.UIFigure);
            app.RGBDropDown.Items = {'Non', 'Full', 'R', 'G', 'B'};
            app.RGBDropDown.ValueChangedFcn = createCallbackFcn(app, @RGBDropDownValueChanged, true);
            app.RGBDropDown.FontName = 'Arial';
            app.RGBDropDown.FontSize = 14;
            app.RGBDropDown.Position = [111 169 100 22];
            app.RGBDropDown.Value = 'Non';

            % Create HSIDropDownLabel
            app.HSIDropDownLabel = uilabel(app.UIFigure);
            app.HSIDropDownLabel.HorizontalAlignment = 'right';
            app.HSIDropDownLabel.FontName = 'Arial';
            app.HSIDropDownLabel.FontSize = 14;
            app.HSIDropDownLabel.FontWeight = 'bold';
            app.HSIDropDownLabel.Position = [343 169 28 22];
            app.HSIDropDownLabel.Text = 'HSI';

            % Create HSIDropDown
            app.HSIDropDown = uidropdown(app.UIFigure);
            app.HSIDropDown.Items = {'Non', 'Full', 'H', 'S', 'I'};
            app.HSIDropDown.ValueChangedFcn = createCallbackFcn(app, @HSIDropDownValueChanged, true);
            app.HSIDropDown.FontName = 'Arial';
            app.HSIDropDown.FontSize = 14;
            app.HSIDropDown.Position = [386 169 100 22];
            app.HSIDropDown.Value = 'Non';

            % Create LabDropDownLabel
            app.LabDropDownLabel = uilabel(app.UIFigure);
            app.LabDropDownLabel.HorizontalAlignment = 'right';
            app.LabDropDownLabel.FontName = 'Arial';
            app.LabDropDownLabel.FontSize = 14;
            app.LabDropDownLabel.FontWeight = 'bold';
            app.LabDropDownLabel.Position = [53 107 41 22];
            app.LabDropDownLabel.Text = 'L*a*b';

            % Create LabDropDown
            app.LabDropDown = uidropdown(app.UIFigure);
            app.LabDropDown.Items = {'Non', 'Full', 'L', 'A', 'B'};
            app.LabDropDown.ValueChangedFcn = createCallbackFcn(app, @LabDropDownValueChanged, true);
            app.LabDropDown.FontName = 'Arial';
            app.LabDropDown.FontSize = 14;
            app.LabDropDown.Position = [109 107 103 22];
            app.LabDropDown.Value = 'Non';

            % Create YCbCrDropDownLabel
            app.YCbCrDropDownLabel = uilabel(app.UIFigure);
            app.YCbCrDropDownLabel.HorizontalAlignment = 'right';
            app.YCbCrDropDownLabel.FontName = 'Arial';
            app.YCbCrDropDownLabel.FontSize = 14;
            app.YCbCrDropDownLabel.FontWeight = 'bold';
            app.YCbCrDropDownLabel.Position = [332 107 49 22];
            app.YCbCrDropDownLabel.Text = 'YCbCr';

            % Create YCbCrDropDown
            app.YCbCrDropDown = uidropdown(app.UIFigure);
            app.YCbCrDropDown.Items = {'Non', 'Full', 'Y', 'Cb', 'Cr'};
            app.YCbCrDropDown.ValueChangedFcn = createCallbackFcn(app, @YCbCrDropDownValueChanged, true);
            app.YCbCrDropDown.FontName = 'Arial';
            app.YCbCrDropDown.FontSize = 14;
            app.YCbCrDropDown.Position = [396 107 100 22];
            app.YCbCrDropDown.Value = 'Non';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Task1_AsyaAbdelFattah__mfile

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end