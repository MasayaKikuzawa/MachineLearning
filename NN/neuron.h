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


//一括学習
void batch_learning(
	vector<teaching_data>& data, //教師データ
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	vector<vector<vector<double> > >& layer_output,//順方向伝搬の際の各ノードの出力
	vector<vector<vector<double> > >& rev_layer_output,//逆伝搬の際の各ノードの出力
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数,
	int input_siz,//入力データの数
	int output_siz,//出力データの数
	double study_rate//学習率
);



//未学習データの値の出力
double non_teaching_data(
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	teaching_data& data,//未学習データ
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数
	int input_siz,//入力データの数(閾値の分も含むので入力データ+1を入力データとしている)
	int output_siz//出力データの数
);



//逐次学習
void sequential_learning(
	vector<teaching_data>& data, //教師データ
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	vector<vector<vector<double> > >& layer_output,//順方向伝搬の際の各ノードの出力
	vector<vector<vector<double> > >& rev_layer_output,//逆伝搬の際の各ノードの出力
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数,
	int input_siz,//入力データの数
	int output_siz,//出力データの数
	double study_rate//学習率
);