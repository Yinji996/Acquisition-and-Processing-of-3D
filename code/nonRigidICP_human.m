function [Final_vertices, solution ] = nonRigidICP_human( Source, Target)

% Set parameters
epsilon = 0.0007;
alphaSet = linspace(40, 1, 10);
gamma = 1;
beTa = 100;

%Draw the Target and template surface using the patch function
figure;
PlotTarget = rmfield(Target, 'normals');
Target_plot = patch(PlotTarget, 'facecolor', [0,0,1], 'EdgeColor',  'none', 'FaceAlpha', 0.5);
hold on;
PlotSource = rmfield(Source, 'normals');
Template_plot = patch(PlotSource, 'facecolor', [1,0,0], 'EdgeColor',  'none','FaceAlpha', 0.5);
material dull; light; grid on; xlabel('x'); ylabel('y'); zlabel('z');
view([23 52]); axis equal; axis manual;
drawnow;
rotate3d;

Source_vertices = Source.vertices;
N_vertices = size(Source_vertices, 1);
Target_vertex = Target.vertices;

% Obtain matrix G 
G = diag([1 1 1 gamma]);

% Calculate node-arc incidence matrix M 
M_tem = triangulation2adjacency(Source.faces, Source.vertices);
M = adjacency2incidence(M_tem)';
% Calculate the kronecker product
kron_M_G = kron(M, G);

% obtain matrix D
v = [Source_vertices,ones(N_vertices,1)];
D = sparse(N_vertices, 4*N_vertices);
for i = 1 : N_vertices
    col = i*4;
    D(i,col-3:col) = v(i,:);
end


% Do rigid ICP first
disp('Project rigid ICP');
[R, t] = icp(Target_vertex', Source_vertices', 50, 'Matching', 'kDtree');
X = repmat([R'; t'], N_vertices, 1);
Final_vertices = D*X;

% Update plot
set(Template_plot, 'Vertices', Final_vertices);
drawnow;

N = sparse(N_vertices, 4*N_vertices);
Source_Normal = Source.normals;
Target_Normal = Target.normals;
new_normal = [Source_Normal,ones(N_vertices,1)];
for i = 1 : N_vertices
    col = i*4;
    N(i,col-3:col) = new_normal(i,:);
end

% Get number of stiffness parameters alphaSet
N_alpha = length(alphaSet);
% Define landmark
[ DL, UL ] = landmarkDefine(kron_M_G,Source_vertices,Target_vertex,'body');         
%scatter3(UL(:,1),UL(:,2),UL(:,3),'g','filled');% sign landmarks
legend('Target', 'Source', 'Target Landmark')
count = 1;

% Non-rigid ICP iteration
for i = 1:N_alpha
    
    % Update stiffness
    alpha =alphaSet(i);
    
    % Set X_k_1 to be very different to X firstly
	X_k_1 = X + 100;
    
    while norm(X - X_k_1) >= epsilon 
        norm(X - X_k_1)
        % Obtain the vertex by current transformation matrix X
        Final_vertices = D*X;
        
        set(Template_plot, 'Vertices', full(Final_vertices));
        drawnow;

        % Find the closest points between target and source
        targetId = knnsearch(Target_vertex, Final_vertices);
        U = Target_vertex(targetId,:);
        
        % Give zero weights to boundary target vertices.
%         tarBoundary = ismember(targetId, boundary_point);
%         weight_vector = ~tarBoundary;
% 
%         Transform_vertex_normal = N*X;
%         Target_vertex_Normal = Target_Normal(targetId,:);
%         weight_vector = largeAngleEliminate(weight_vector,Target_vertex_Normal,Transform_vertex_normal);

        weight_vector = ones(N_vertices,1);
            
        % Update weight matrix
        W = spdiags(weight_vector, 0, N_vertices, N_vertices);

        % Generate the matrix A and B
        A = [alpha .* kron_M_G; W * D; beTa .* DL];
        B = [zeros(size(M,1)*size(G,1), 3); W * U; beTa .* UL];
        
        % Get optimal transformation X and remember old transformation X_k_1
        X_k_1 = X;
        X = (A' * A) \ (A' * B);
        solution.error(count) = norm(Final_vertices - U);
        count = count + 1;
    end
end

% Compute transformed points 
Final_vertices = D*X;

%process the rigid ICP 
disp('project rigid ICP');
[R, t] = icp(Target_vertex', Final_vertices', 50, 'Matching', 'kDtree');
for i = 1:N_vertices
    Final_vertices(i,:) = Final_vertices(i,:) * R + t';
end

set(Template_plot, 'Vertices', full(Final_vertices));
drawnow;

Target_Id = knnsearch(Target_vertex, Final_vertices);
U_final = Target_vertex(Target_Id,:);
solution.error(count) = norm(Final_vertices - U_final);