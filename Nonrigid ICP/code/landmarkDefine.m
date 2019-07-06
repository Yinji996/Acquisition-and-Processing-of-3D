function [ DL, UL ] = landmarkDefine(kron_M_G,vertsSource,vertsTarget, type)

% landmark points
switch type
    case'face'
        landmarkSource = [3048 3051 1754 4112 4122 1030 1032 4118 4118 4768 4768 4506 4506];
        landmarkTarget = [2984 2986 1773 4056 4075 1105 1106 4295 4296 4831 4833 4486 4485];
    case'body'
        landmarkSource = [1:20:5100 1600:1700 5033:5233 332 337 386 1863 5326 1700 5175 2674 2657 2446 2320 2747 6760:10:6880 ...
                  6134 6017 5906 5783 6207 3501 2104 5560 1044 4530 3217 6617 1305 1306 4219 4220 1247 1248 6527 6528 ...
                  2300:2800 5700:6200];
        landmarkTarget = [1:20:5100 1600:1700 5033:5233 332 337 386 1863 5326 1700 5175 2674 2657 2446 2320 2747 6760:10:6880 ...
                          6134 6017 5906 5783 6207 3501 2104 5560 1044 4530 3217 6617 1305 1306 4219 4220 1247 1248 6527 6528 ...
                          2300:2800 5700:6200];
end
                      
landmarkNumber = size(landmarkSource,2);
DL = zeros(landmarkNumber,size(kron_M_G,2));
UL = zeros(landmarkNumber,3);

% assign value
for tem = 1:landmarkNumber
    currentSource = landmarkSource(tem);
    currentTarget = landmarkTarget(tem);
    
    DL(tem,4 * currentSource - 3:4 * currentSource - 1) = vertsSource(currentSource,1:3);
    DL(tem,4 * currentSource) = 1;
    UL(tem,1:3) = vertsTarget(currentTarget,1:3);
end