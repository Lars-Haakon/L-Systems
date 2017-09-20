#ifndef MESH_H
#define MESH_H

class Mesh
{
public:
	Mesh(float* vertices, int n_vertices, int* indices, int n_indices);
	~Mesh();

	void Draw();
private:
	unsigned int m_vbo;
	unsigned int m_ibo;
	int m_size;
};

#endif
