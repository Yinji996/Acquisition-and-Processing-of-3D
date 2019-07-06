%% Nonrigid ICP algorithm
%add the library function
addpath('..\Library\Iterative Closest Point')
addpath('..\Library\toolbox_graph')

%% Load foundamental data
addpath('..\data');
%load the source dataset
[Source_Vertex,Source_Face] = read_off('CaixuKun5000.off');
%compute the vertex normal
[Source_Normal,~] = compute_normal(Source_Vertex,Source_Face);

Source_Face = Source_Face';
Source.faces = Source_Face;
Source_Vertex = Source_Vertex';
Source.vertices = Source_Vertex/10000;
Source.normals = Source_Normal';

% Load Target dataset
[Target_Vertex,Target_Face] = read_off('LiuziChen5000.off');
%compute the vertex normal
[Target_Normal,~] = compute_normal(Target_Vertex,Target_Face);

Target_Face = Target_Face';
Target.faces = Target_Face;
Target_Vertex = Target_Vertex';
Target.vertices = Target_Vertex/10000;
Target.normals = Target_Normal';

%% load extended source
% Load source dataset
[Source_Vertex_h,Source_Face_h] = read_off('human_source.off');
%compute the vertex normal
[Source_Normal_h,~] = compute_normal(Source_Vertex_h,Source_Face_h);
Source_Face_h = Source_Face_h';
Source_h.faces = Source_Face_h;
Source_Vertex_h = Source_Vertex_h';
Source_h.vertices = Source_Vertex_h;
Source_h.normals = Source_Normal_h';


% Load Target dataset
[Target_Vertex_h,Target_Face_h] = read_off('human_target.off');
%compute the vertex normal
[Target_Normal_h,~] = compute_normal(Target_Vertex_h,Target_Face_h);

Target_Face_h = Target_Face_h';
Target_h.faces = Target_Face_h;
Target_Vertex_h = Target_Vertex_h';
Target_h.vertices = Target_Vertex_h;
Target_h.normals = Target_Normal_h';

%% Non-rigid ICP 
[find_vertex, solution] = nonRigidICP(Source, Target);
[find_vertex_human, solution_human] = nonRigidICP_human(Source_h, Target_h);
