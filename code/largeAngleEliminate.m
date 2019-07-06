function [weight_vector] = largeAngleEliminate(weight_vector,Target_vertex_Normal,Transform_vertex_normal)

Cross_normals = cross(Target_vertex_Normal, Transform_vertex_normal);
Cross_normalsNorm = sqrt(sum(Cross_normals.^2,2));
dotNormals = dot(Target_vertex_Normal, Transform_vertex_normal, 2);
angle = atan2(Cross_normalsNorm, dotNormals);
weight_vector = weight_vector .* (angle<pi/4);

end

