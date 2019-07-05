#include <iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<random>


#include"teaching_data.h";
#include"macro.h";
using namespace std;
const double eps = 1e-8;//�덷臒l
const int rep = 30000;//�w�K��


/*****************************************************************************************************/
//�v���g�^�C�v�錾
double sigmoid(double gain,double s);//�V�O���C�h�֐�
void value_initialize(vector<vector<vector<double> > >& weight);//�d�݌W���̏�����

//�ꊇ�w�K
void batch_learning(
	vector<teaching_data>& data, //���t�f�[�^
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	vector<vector<vector<double> > >& layer_output,//�������`���̍ۂ̊e�m�[�h�̏o��
	vector<vector<vector<double> > >& rev_layer_output,//�t�`���̍ۂ̊e�m�[�h�̏o��
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��
	int input_siz,//���̓f�[�^�̐�(臒l�̕����܂ނ̂œ��̓f�[�^+1����̓f�[�^�Ƃ��Ă���)
	int output_siz,//�o�̓f�[�^�̐�
	double study_rate//�w�K��
);


//�����w�K
void sequential_learning(
	vector<teaching_data>& data, //���t�f�[�^
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	vector<vector<vector<double> > >& layer_output,//�������`���̍ۂ̊e�m�[�h�̏o��
	vector<vector<vector<double> > >& rev_layer_output,//�t�`���̍ۂ̊e�m�[�h�̏o��
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��
	int input_siz,//���̓f�[�^�̐�(臒l�̕����܂ނ̂œ��̓f�[�^+1����̓f�[�^�Ƃ��Ă���)
	int output_siz,//�o�̓f�[�^�̐�
	double study_rate//�w�K��
);


//���w�K�f�[�^�̒l�̏o��
double non_teaching_data(
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	teaching_data& data,//���w�K�f�[�^
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��
	int input_siz,//���̓f�[�^�̐�(臒l�̕����܂ނ̂œ��̓f�[�^+1����̓f�[�^�Ƃ��Ă���)
	int output_siz//�o�̓f�[�^�̐�
);

/****************************************************************************************************/



//�ꊇ�w�K
void batch_learning(
	vector<teaching_data>& data, //���t�f�[�^
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	vector<vector<vector<double> > >& layer_output,//�������`���̍ۂ̊e�m�[�h�̏o��
	vector<vector<vector<double> > >& rev_layer_output,//�t�`���̍ۂ̊e�m�[�h�̏o��
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��
	int input_siz,//���̓f�[�^�̐�(臒l�̕����܂ނ̂œ��̓f�[�^+1����̓f�[�^�Ƃ��Ă���)
	int output_siz,//�o�̓f�[�^�̐�
	double study_rate//�w�K��
) {

	//�z��̓Y��������
	//weight[i][j][k]:=i�w�̃m�[�hj��i+1�w�̃m�[�hk�̊Ԃ̏d�݌W��
	//layer_out[i][j][k]:= i�Ԗڂ̃f�[�^�ɑ΂��鏇�����`���̍ۂ�j�w�̃m�[�hk�̏o��
	//rev_layer_out[i][j][k]:= i�Ԗڂ̃f�[�^�ɑ΂���t�����`���̍ۂ�j�w�̃m�[�hk�̏o��

	/***************************************/
	//�g���ϐ�
	int all_layer_siz = middle_layer_siz + 2;//���ԑw + ���͑w + �o�͑w
	int data_siz = data.size();//���t�f�[�^�̐�
	auto sum_u = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//�������`���̍ۂ̃V�O���C�h�֐��ɓ����O�̘a
	auto sum_z = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//�t�����`���̍ۂ̏o�͒l
	auto diff = make_v<double>(data_siz + 10);

	/********************************************/


	//�����������Əd�݂̗����ł̏�����

	value_initialize(weight);//�d�݂̏�����

	/********************************************/

	
	//�z���1-index

	double error = 0;//�덷(�����l��1.0)
	int cnt = 0;//�w�K��

	do//�덷��臒l�ȉ��ɂȂ邩�w�K�񐔂̏���ɒB����܂ł͊w�K���p������
	{
		error = 0.0;
		//�������`������
		//�덷��error�ɑ������


		fill_v(sum_u, 0);//�������o��
		fill_v(sum_z, 0);//�t�����o��
		fill_v(diff, 0);//�덷

		//***********************************************************************************************
		//�ȉ��������`��

		for (int data_i = 0; data_i < data_siz; data_i++)//�f�[�^��S�ē��͂��Č덷�����v����
		{
			//cout << "data_i = " << data_i << endl;
			for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)//all_layer_siz�͏o�͑w�̃C���f�b�N�X�ƈ�v���Ă���
			{
				//�S�Ă̑w�ɂ��ē`��������
				//layer_num�w��layer_num+1�w�̓`���ɂ���
				//���͑w�ł͎��̑w�ɓ`��������
				//���͑w�ȊO�͑O�̑w��sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂Ă��玟�̑w�ɓ`��������


				if (layer_num == all_layer_siz)
				{
					//�o�͑w�̏ꍇ
					//�܂�sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂�
					layer_output[data_i][layer_num][1] = sigmoid(1.0, sum_u[data_i][layer_num][1]);
					error += pow(layer_output[data_i][layer_num][1] - data[data_i].output[0], 2) / 2;//�]���֐�
					diff[data_i] = layer_output[data_i][layer_num][1] - data[data_i].output[0];//�덷	
				}
				else
				{
					//�o�͑w�ȊO�̏���

					if (layer_num == 1)
					{
						//���͑w�̓`���O�̏���
						//�o�͒l�̐ݒ�
						//���͑w�̓��͂͂��̂܂܏o�͂���(0��臒l,���̓f�[�^��1-index�ɂȂ��Ă���)
						for (int now_node = 0; now_node < input_siz; now_node++)
						{
							//���͑w�̃f�[�^�̏�����
							layer_output[data_i][layer_num][now_node] = data[data_i].input[now_node];
						}


						//layer_num+1�w�Ɋe�o�͒l��`��������
						for (int now_node = 0; now_node < input_siz; now_node++)
						{

							//�`�����̃m�[�h��臒l���܂߂đS�ē`��������
							//�o�͑w�ւ̓`���̕����Ńm�[�h1�ȊO�̃m�[�h�ɂ��`�������邱�ƂɂȂ邪�C���̃m�[�h�̒l���g�����Ƃ͂Ȃ����߁C���̏����Ŗ��͂Ȃ�
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//�`����̃m�[�h��臒l���܂߂Ȃ�
								//�e�w�̑f�q�����S�Ăɓ`��������
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}
					}
					else
					{
						//���ԑw�̓`���O�̏���
						//sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂�
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							layer_output[data_i][layer_num][now_node] = sigmoid(1.0, sum_u[data_i][layer_num][now_node]);
						}

						//layer_num+1�w�Ɋe�o�͒l��`��������
						for (int now_node = 0; now_node <= each_layer_siz; now_node++)
						{

							//�`�����̃m�[�h��臒l���܂߂đS�ē`��������
							//�o�͑w�ւ̓`���̕����Ńm�[�h1�ȊO�̃m�[�h�ɂ��`�������邱�ƂɂȂ邪�C���̃m�[�h�̒l���g�����Ƃ͂Ȃ����߁C���̏����Ŗ��͂Ȃ�
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//�`����̃m�[�h��臒l���܂߂Ȃ�
								//�e�w�̑f�q�����S�Ăɓ`��������
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}

					}
				}
			}
		}



		//�S�Ẵf�[�^�ɂ��Č덷�̍��v�����߂�̂��I������
		//****************************************************************************************************************

		//�ȉ��덷�S�ʂ��s������




		for (int layer_num = all_layer_siz; layer_num > 1; layer_num--)//all_layer_siz�͏o�͑w�̃C���f�b�N�X�ƈ�v���Ă���
		{
			//�S�Ă̑w�ɂ�2���ē`��������
			//layer_num�w��layer_num-1�w�̓`���ɂ���
			//�o�͑w�ł͎��̑w�ɓ`��������
			//�o�͑w�ȊO�͑O�̑w��sum_z�Ɍ��݂̑w�̏o��y�ɑ΂���y(1-y)��������O���������Ă��玟�̑w�֓`��������

			if (layer_num == all_layer_siz)
			{
				//�o�͑w�̏ꍇ
				//�`��������
				for (int next_node = 0; next_node <= each_layer_siz; next_node++)
				{
					//�`�������镔��
					//�O�̑w�ւ̋t�`����(��̑w����̓���)*(���݂̑w�̏o��)*(1-���݂̑w�̏o��)*(�d�݌W��)

					for (int data_i = 0; data_i < data_siz; data_i++)
					{
						sum_z[data_i][layer_num - 1][next_node] += (diff[data_i]) * layer_output[data_i][layer_num][1]
							* (1 - layer_output[data_i][layer_num][1]) * weight[layer_num - 1][next_node][1];
					}
				}
				//�d�݌W���̍X�V(�`�d��)
				for (int next_node = 0; next_node <= each_layer_siz; next_node++)
				{
					//�O�̑w�Ƃ̊Ԃ̏d�݂̍X�V��(��̑w����̓���)*(���݂̑w�̏o��)*(1-���݂̑w�̏o��)*(�O�̑w�̏o��)
					double tmp = 0;

					for (int data_i = 0; data_i < data_siz; data_i++)
					{
						//�S�Ẵf�[�^�ɂ��č��v
						tmp += (diff[data_i]) * layer_output[data_i][layer_num - 1][next_node]
							* layer_output[data_i][layer_num][1] * (1 - layer_output[data_i][layer_num][1]);
					}
					tmp /= data_siz;//���ς��Ƃ���

					//�d�݂��X�V
					weight[layer_num - 1][next_node][1]
						= weight[layer_num - 1][next_node][1] - study_rate * tmp;
				}
			}
			else
			{
				//�o�͑w�ȊO�̏���
				//�����ł͂܂��C���݂̑w��sum_z�Ɍ��݂̑w�̏o��layer_output�ɑ΂���(1-layer_output)*layer_output��������O�������s��



				if (layer_num == 2)
				{
					//���͑w��1�O�̑w
					//�ꊇ�w�K�ł͑S�Ẵf�[�^�̌덷�����v���ċt�`���̃t�F�[�Y�Ɉڂ邽�߁C�����ł͌덷�����Z���ďI��
					//�`�d�����镔��
					//�����ł͏d�݌W���������ē`��������
					for (int now_node = 1; now_node <= each_layer_siz; now_node++)
					{
						for (int next_node = 0; next_node < input_siz; next_node++)
						{

							for (int data_i = 0; data_i < data_siz; data_i++)
							{
								sum_z[data_i][layer_num - 1][next_node] += sum_z[data_i][layer_num][now_node] * weight[layer_num - 1][next_node][now_node]
									* layer_output[data_i][layer_num][now_node] * (1 - layer_output[data_i][layer_num][now_node]);
							}
						}
					}

					//�d�݌W���̍X�V
					for (int now_node = 1; now_node <= each_layer_siz; now_node++)
					{
						for (int next_node = 0; next_node < input_siz; next_node++)
						{


							double tmp = 0;

							for (int data_i = 0; data_i < data_siz; data_i++)
							{
								//�S�Ẵf�[�^�ɂ��č��v
								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);
							}
							tmp /= data_siz;//���ς��Ƃ���

							//�d�݂̍X�V
							weight[layer_num - 1][next_node][now_node]
								= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;
						}
					}
				}
				else
				{
					//���ԑw


					//�`�d�����镔��
					//�����ł͏d�݌W���������ē`��������
					for (int now_node = 1; now_node <= each_layer_siz; now_node++)
					{
						for (int next_node = 0; next_node <= each_layer_siz; next_node++)
						{

							for (int data_i = 0; data_i < data_siz; data_i++)
							{
								sum_z[data_i][layer_num - 1][next_node] += sum_z[data_i][layer_num][now_node] * weight[layer_num - 1][next_node][now_node]
									* layer_output[data_i][layer_num][now_node] * (1 - layer_output[data_i][layer_num][now_node]);
							}
						}
					}

					//�d�݌W���̍X�V
					for (int now_node = 1; now_node <= each_layer_siz; now_node++)
					{
						for (int next_node = 0; next_node <= each_layer_siz; next_node++)
						{

							double tmp = 0;

							for (int data_i = 0; data_i < data_siz; data_i++)
							{
								//�S�Ẵf�[�^�ɂ��č��v
								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);
							}
							tmp /= data_siz;//���ς��Ƃ���

							//�d�݂̍X�V
							weight[layer_num - 1][next_node][now_node]
								= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;

						}
					}
				}
			}
		}

		//�w�K�񐔂��X�V��Y��Ȃ�
		cnt++;

		cout << "cnt = " << cnt << " error= " << error << endl;

	} while (abs(error) > eps && cnt < rep);//�덷��臒l�ȉ��ɂȂ邩�w�K�񐔂̏���ɒB����܂ł͊w�K���p������

	return;
}



//�����w�K
void sequential_learning(
	vector<teaching_data>& data, //���t�f�[�^
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	vector<vector<vector<double> > >& layer_output,//�������`���̍ۂ̊e�m�[�h�̏o��
	vector<vector<vector<double> > >& rev_layer_output,//�t�`���̍ۂ̊e�m�[�h�̏o��
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��
	int input_siz,//���̓f�[�^�̐�(臒l�̕����܂ނ̂œ��̓f�[�^+1����̓f�[�^�Ƃ��Ă���)
	int output_siz,//�o�̓f�[�^�̐�
	double study_rate//�w�K��
) {

	//�z��̓Y��������
	//weight[i][j][k]:=i�w�̃m�[�hj��i+1�w�̃m�[�hk�̊Ԃ̏d�݌W��
	//layer_out[i][j][k]:= i�Ԗڂ̃f�[�^�ɑ΂��鏇�����`���̍ۂ�j�w�̃m�[�hk�̏o��
	//rev_layer_out[i][j][k]:= i�Ԗڂ̃f�[�^�ɑ΂���t�����`���̍ۂ�j�w�̃m�[�hk�̏o��


	//�f�o�b�O�p

	/***************************************/
	//�g���ϐ�
	int all_layer_siz = middle_layer_siz + 2;//���ԑw + ���͑w + �o�͑w
	int data_siz = data.size();//���t�f�[�^�̐�
	auto sum_u = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//�������`���̍ۂ̃V�O���C�h�֐��ɓ����O�̘a
	auto sum_z = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//�t�����`���̍ۂ̏o�͒l
	auto diff = make_v<double>(data_siz + 10);

	/********************************************/


	//�����������Əd�݂̗����ł̏�����

	value_initialize(weight);//�d�݂̏�����

	/********************************************/
	//�z���1-index

	double error = 0;//�덷(�����l��1.0)
	int cnt = 0;//�w�K��

	do//�덷��臒l�ȉ��ɂȂ邩�w�K�񐔂̏���ɒB����܂ł͊w�K���p������
	{
		error = 0.0;
		//�������`������
		//�덷��error�ɑ������


		fill_v(sum_u, 0);//�������o��
		fill_v(sum_z, 0);//�t�����o��
		fill_v(diff, 0);//�덷

		//***********************************************************************************************
		//�ȉ��������`��

		for (int data_i = 0; data_i < data_siz; data_i++)//�f�[�^��S�ē��͂��Č덷�����v����
		{
			for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)//all_layer_siz�͏o�͑w�̃C���f�b�N�X�ƈ�v���Ă���
			{
				//�S�Ă̑w�ɂ��ē`��������//
				//layer_num�w��layer_num+1�w�̓`���ɂ���
				//���͑w�ł͎��̑w�ɓ`��������
				//���͑w�ȊO�͑O�̑w��sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂Ă��玟�̑w�ɓ`��������


				if (layer_num == all_layer_siz)
				{
					//�o�͑w�̏ꍇ
					layer_output[data_i][layer_num][1] = sigmoid(1.0, sum_u[data_i][layer_num][1]);
					
					error += pow(layer_output[data_i][layer_num][1] - data[data_i].output[0], 2) / 2;//�]���֐�
					diff[data_i] = layer_output[data_i][layer_num][1] - data[data_i].output[0];//�덷	

				}
				else
				{
					//�o�͑w�ȊO�̏���

					if (layer_num == 1)
					{
						//���͑w�̓`���O�̏���
						//�o�͒l�̐ݒ�
						//���͑w�̓��͂͂��̂܂܏o�͂���(0��臒l,���̓f�[�^��1-index�ɂȂ��Ă���)
						for (int now_node = 0; now_node < input_siz; now_node++)
						{
							//���͑w�̃f�[�^�̏�����
							layer_output[data_i][layer_num][now_node] = data[data_i].input[now_node];
						}


						//layer_num+1�w�Ɋe�o�͒l��`��������
						for (int now_node = 0; now_node < input_siz; now_node++)
						{

							//�`�����̃m�[�h��臒l���܂߂đS�ē`��������
							//�o�͑w�ւ̓`���̕����Ńm�[�h1�ȊO�̃m�[�h�ɂ��`�������邱�ƂɂȂ邪�C���̃m�[�h�̒l���g�����Ƃ͂Ȃ����߁C���̏����Ŗ��͂Ȃ�
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//�`����̃m�[�h��臒l���܂߂Ȃ�
								//�e�w�̑f�q�����S�Ăɓ`��������
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}
					}
					else
					{
						//���ԑw�̓`���O�̏���
						//sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂�
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							layer_output[data_i][layer_num][now_node] = sigmoid(1.0, sum_u[data_i][layer_num][now_node]);
							//cout << "layer_num " << layer_num << " now_node " << now_node << " before =" << sum_u[layer_num][now_node] << " after = " << layer_output[layer_num][now_node] << endl;
						}

						//layer_num+1�w�Ɋe�o�͒l��`��������
						for (int now_node = 0; now_node <= each_layer_siz; now_node++)
						{

							//�`�����̃m�[�h��臒l���܂߂đS�ē`��������
							//�o�͑w�ւ̓`���̕����Ńm�[�h1�ȊO�̃m�[�h�ɂ��`�������邱�ƂɂȂ邪�C���̃m�[�h�̒l���g�����Ƃ͂Ȃ����߁C���̏����Ŗ��͂Ȃ�
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//�`����̃m�[�h��臒l���܂߂Ȃ�
								//�e�w�̑f�q�����S�Ăɓ`��������
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}

					}
				}
			}


			//�����Ō덷�t�`��

			for (int layer_num = all_layer_siz; layer_num > 1; layer_num--)//all_layer_siz�͏o�͑w�̃C���f�b�N�X�ƈ�v���Ă���
			{
				//�S�Ă̑w�ɂ�2���ē`��������
				//layer_num�w��layer_num-1�w�̓`���ɂ���
				//�o�͑w�ł͎��̑w�ɓ`��������
				//�o�͑w�ȊO�͑O�̑w��sum_z�Ɍ��݂̑w�̏o��y�ɑ΂���y(1-y)��������O���������Ă��玟�̑w�֓`��������

				if (layer_num == all_layer_siz)
				{
					//�o�͑w�̏ꍇ
					//�`��������
					for (int next_node = 0; next_node <= each_layer_siz; next_node++)
					{
						//�`�������镔��
						//�O�̑w�ւ̋t�`����(��̑w����̓���)*(���݂̑w�̏o��)*(1-���݂̑w�̏o��)*(�d�݌W��)

						sum_z[data_i][layer_num - 1][next_node] += (diff[data_i]) * layer_output[data_i][layer_num][1]
							* (1 - layer_output[data_i][layer_num][1]) * weight[layer_num - 1][next_node][1];
					}

					//cout << endl;
					//�d�݌W���̍X�V(�`�d��)
					for (int next_node = 0; next_node <= each_layer_siz; next_node++)
					{
						//�O�̑w�Ƃ̊Ԃ̏d�݂̍X�V��(��̑w����̓���)*(���݂̑w�̏o��)*(1-���݂̑w�̏o��)*(�O�̑w�̏o��)
						double tmp = 0;

						tmp += (diff[data_i]) * layer_output[data_i][layer_num - 1][next_node]
							* layer_output[data_i][layer_num][1] * (1 - layer_output[data_i][layer_num][1]);

						//�d�݂��X�V
						weight[layer_num - 1][next_node][1]
							= weight[layer_num - 1][next_node][1] - study_rate * tmp;
					}
				}
				else
				{
					//�o�͑w�ȊO�̏���
					//�����ł͂܂��C���݂̑w��sum_z�Ɍ��݂̑w�̏o��layer_output�ɑ΂���(1-layer_output)*layer_output��������O�������s��

					if (layer_num == 2)
					{
						//���͑w��1�O�̑w
						//�ꊇ�w�K�ł͑S�Ẵf�[�^�̌덷�����v���ċt�`���̃t�F�[�Y�Ɉڂ邽�߁C�����ł͌덷�����Z���ďI��
						//�`�d�����镔��
						//�����ł͏d�݌W���������ē`��������
						//cout << "layer_num =" << layer_num << endl;
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node < input_siz; next_node++)
							{
								sum_z[data_i][layer_num - 1][next_node] += sum_z[data_i][layer_num][now_node] * weight[layer_num - 1][next_node][now_node]
									* layer_output[data_i][layer_num][now_node] * (1 - layer_output[data_i][layer_num][now_node]);
							}
						}

						//�d�݌W���̍X�V
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node < input_siz; next_node++)
							{


								double tmp = 0;

								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);

								//�d�݂̍X�V
								weight[layer_num - 1][next_node][now_node]
									= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;
							}
						}
					}
					else
					{
						//���ԑw


						//�`�d�����镔��
						//�����ł͏d�݌W���������ē`��������
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node <= each_layer_siz; next_node++)
							{
								sum_z[data_i][layer_num - 1][next_node] += sum_z[data_i][layer_num][now_node] * weight[layer_num - 1][next_node][now_node]
									* layer_output[data_i][layer_num][now_node] * (1 - layer_output[data_i][layer_num][now_node]);
							}
						}

						//�d�݌W���̍X�V
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node <= each_layer_siz; next_node++)
							{

								double tmp = 0;

								//�S�Ẵf�[�^�ɂ��č��v
								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);

								//�d�݂̍X�V
								weight[layer_num - 1][next_node][now_node]
									= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;

							}
						}
					}
				}

			}
		}
		//�w�K�񐔂��X�V��Y��Ȃ�
		cnt++;

		cout << "cnt = " << cnt << " error= " << error << endl;

	} while (abs(error) > eps && cnt < rep);//�덷��臒l�ȉ��ɂȂ邩�w�K�񐔂̏���ɒB����܂ł͊w�K���p������


	return;
}



//�V�O���C�h�֐�(�Q�C����1)
double sigmoid(double gain,double s) {

	return 1.0 / (1.0 + exp(-gain * s));

}



//�d�݌W�������肷��֐�
void value_initialize(vector<vector<vector<double> > >& weight) {

	// 0.0�ȏ�1.0�����̒l�𓙊m���Ŕ���������
	std::random_device rnd;     // �񌈒�I�ȗ���������𐶐�
	std::mt19937 mt(rnd());     //  �����Z���k�E�c�C�X�^��32�r�b�g�ŁA�����͏����V�[�h�l
	std::uniform_real_distribution<> rand12(-1.0, 1.0);   

	for (int i = 0; i < weight.size(); i++) {

		for (int j = 0; j < weight[i].size(); j++)
		{
			for (int k = 0; k < weight[i][j].size(); k++)
			{
				weight[i][j][k] = rand12(mt);
			}
		}
	}
}



//���w�K�f�[�^�̒l�̏o��
double non_teaching_data(
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	teaching_data& data,//���w�K�f�[�^
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��
	int input_siz,//���̓f�[�^�̐�(臒l�̕����܂ނ̂œ��̓f�[�^+1����̓f�[�^�Ƃ��Ă���)
	int output_siz//�o�̓f�[�^�̐�
) {

	//�����ɏ���

	int all_layer_siz = middle_layer_siz + 2;//���ԑw + ���͑w + �o�͑w
	auto sum_u = make_v<double>(all_layer_siz + 10, all_layer_siz + 10);//�������`���̍ۂ̃V�O���C�h�֐��ɓ����O�̘a
	fill_v(sum_u, 0);
	auto layer_output = make_v<double>(middle_layer_siz + 10, each_layer_siz + 10);//�������`���̍ۂ̊e�m�[�h�̏o�͒l�p
	fill_v(layer_output, 0);

	for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)
	{
		//臒l�̕����̏�����
		layer_output[layer_num][0] = 1.0;
	}

	for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)//all_layer_siz�͏o�͑w�̃C���f�b�N�X�ƈ�v���Ă���
	{
		//�S�Ă̑w�ɂ��ē`��������
		//layer_num�w��layer_num+1�w�̓`���ɂ���
		//���͑w�ł͎��̑w�ɓ`��������
		//���͑w�ȊO�͑O�̑w��sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂Ă��玟�̑w�ɓ`��������


		if (layer_num == all_layer_siz)
		{
			//�o�͑w�̏ꍇ
			//�܂�sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂�
			layer_output[layer_num][1] = sigmoid(1, sum_u[layer_num][1]);
			cout << "sum_u = " << sum_u[layer_num][1] << endl;
			return layer_output[layer_num][1];

		}
		else
		{
			//�o�͑w�ȊO�̏���

			if (layer_num == 1)
			{
				//���͑w�̓`���O�̏���
				//�o�͒l�̐ݒ�
				//�ꊇ�w�K�ł͑S�Ẵf�[�^�̌덷�����v���ċt�`���̃t�F�[�Y�Ɉڂ邽�߁C�����ł͌덷�����Z���ďI��


				//���͑w�̓��͂͂��̂܂܏o�͂���(���̓f�[�^��0-index�ɂȂ��Ă���)
				for (int now_node = 0; now_node < input_siz; now_node++)
				{
					layer_output[1][now_node] = data.input[now_node];
				}


				//layer_num+1�w�Ɋe�o�͒l��`��������
				for (int now_node = 0; now_node < input_siz; now_node++)
				{

					//�`�����̃m�[�h��臒l���܂߂đS�ē`��������
					//�o�͑w�ւ̓`���̕����Ńm�[�h1�ȊO�̃m�[�h�ɂ��`�������邱�ƂɂȂ邪�C���̃m�[�h�̒l���g�����Ƃ͂Ȃ����߁C���̏����Ŗ��͂Ȃ�
					for (int next_node = 1; next_node <= each_layer_siz; next_node++)
					{
						//�`����̃m�[�h��臒l���܂߂Ȃ�
						//�e�w�̑f�q�����S�Ăɓ`��������
						sum_u[layer_num + 1][next_node] += layer_output[layer_num][now_node] * weight[layer_num][now_node][next_node];
					}
				}
			}
			else
			{
				//���ԑw�̓`���O�̏���
				//sum_u���V�O���C�h�֐��ɑ�����ďo�͒l�����߂�
				for (int now_node = 1; now_node <= each_layer_siz; now_node++)
				{
					layer_output[layer_num][now_node] = sigmoid(1, sum_u[layer_num][now_node]);
				}

				//layer_num+1�w�Ɋe�o�͒l��`��������
				for (int now_node = 0; now_node <= each_layer_siz; now_node++)
				{

					//�`�����̃m�[�h��臒l���܂߂đS�ē`��������
					//�o�͑w�ւ̓`���̕����Ńm�[�h1�ȊO�̃m�[�h�ɂ��`�������邱�ƂɂȂ邪�C���̃m�[�h�̒l���g�����Ƃ͂Ȃ����߁C���̏����Ŗ��͂Ȃ�
					for (int next_node = 1; next_node <= each_layer_siz; next_node++)
					{
						//�`����̃m�[�h��臒l���܂߂Ȃ�
						//�e�w�̑f�q�����S�Ăɓ`��������
						sum_u[layer_num + 1][next_node] += layer_output[layer_num][now_node] * weight[layer_num][now_node][next_node];
					}
				}

			}
		}
	}
}
