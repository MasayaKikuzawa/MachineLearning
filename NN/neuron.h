#pragma once
#include <iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>

#include"in_out.h"
#include"macro.h";
#include"teaching_data.h";
using namespace std;


//�ꊇ�w�K
void batch_learning(
	vector<teaching_data>& data, //���t�f�[�^
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	vector<vector<vector<double> > >& layer_output,//�������`���̍ۂ̊e�m�[�h�̏o��
	vector<vector<vector<double> > >& rev_layer_output,//�t�`���̍ۂ̊e�m�[�h�̏o��
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��,
	int input_siz,//���̓f�[�^�̐�
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



//�����w�K
void sequential_learning(
	vector<teaching_data>& data, //���t�f�[�^
	vector<vector<vector<double> > >& weight,//�e�m�[�h�Ԃ̏d��
	vector<vector<vector<double> > >& layer_output,//�������`���̍ۂ̊e�m�[�h�̏o��
	vector<vector<vector<double> > >& rev_layer_output,//�t�`���̍ۂ̊e�m�[�h�̏o��
	int middle_layer_siz,//���ԑw�̐�
	int each_layer_siz,//�e�w�̑f�q��,
	int input_siz,//���̓f�[�^�̐�
	int output_siz,//�o�̓f�[�^�̐�
	double study_rate//�w�K��
);