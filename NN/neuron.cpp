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
const double eps = 1e-8;//誤差閾値
const int rep = 30000;//学習回数


/*****************************************************************************************************/
//プロトタイプ宣言
double sigmoid(double gain,double s);//シグモイド関数
void value_initialize(vector<vector<vector<double> > >& weight);//重み係数の初期化

//一括学習
void batch_learning(
	vector<teaching_data>& data, //教師データ
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	vector<vector<vector<double> > >& layer_output,//順方向伝搬の際の各ノードの出力
	vector<vector<vector<double> > >& rev_layer_output,//逆伝搬の際の各ノードの出力
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数
	int input_siz,//入力データの数(閾値の分も含むので入力データ+1を入力データとしている)
	int output_siz,//出力データの数
	double study_rate//学習率
);


//逐次学習
void sequential_learning(
	vector<teaching_data>& data, //教師データ
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	vector<vector<vector<double> > >& layer_output,//順方向伝搬の際の各ノードの出力
	vector<vector<vector<double> > >& rev_layer_output,//逆伝搬の際の各ノードの出力
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数
	int input_siz,//入力データの数(閾値の分も含むので入力データ+1を入力データとしている)
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

/****************************************************************************************************/



//一括学習
void batch_learning(
	vector<teaching_data>& data, //教師データ
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	vector<vector<vector<double> > >& layer_output,//順方向伝搬の際の各ノードの出力
	vector<vector<vector<double> > >& rev_layer_output,//逆伝搬の際の各ノードの出力
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数
	int input_siz,//入力データの数(閾値の分も含むので入力データ+1を入力データとしている)
	int output_siz,//出力データの数
	double study_rate//学習率
) {

	//配列の添え字メモ
	//weight[i][j][k]:=i層のノードjとi+1層のノードkの間の重み係数
	//layer_out[i][j][k]:= i番目のデータに対する順方向伝搬の際のj層のノードkの出力
	//rev_layer_out[i][j][k]:= i番目のデータに対する逆方向伝搬の際のj層のノードkの出力

	/***************************************/
	//使う変数
	int all_layer_siz = middle_layer_siz + 2;//中間層 + 入力層 + 出力層
	int data_siz = data.size();//教師データの数
	auto sum_u = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//順方向伝搬の際のシグモイド関数に入れる前の和
	auto sum_z = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//逆方向伝搬の際の出力値
	auto diff = make_v<double>(data_siz + 10);

	/********************************************/


	//乱数初期化と重みの乱数での初期化

	value_initialize(weight);//重みの初期化

	/********************************************/

	
	//配列は1-index

	double error = 0;//誤差(初期値は1.0)
	int cnt = 0;//学習回数

	do//誤差が閾値以下になるか学習回数の上限に達するまでは学習を継続する
	{
		error = 0.0;
		//順方向伝搬部分
		//誤差はerrorに代入する


		fill_v(sum_u, 0);//順方向出力
		fill_v(sum_z, 0);//逆方向出力
		fill_v(diff, 0);//誤差

		//***********************************************************************************************
		//以下順方向伝搬

		for (int data_i = 0; data_i < data_siz; data_i++)//データを全て入力して誤差を合計する
		{
			//cout << "data_i = " << data_i << endl;
			for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)//all_layer_sizは出力層のインデックスと一致している
			{
				//全ての層について伝搬させる
				//layer_num層→layer_num+1層の伝搬について
				//入力層では次の層に伝搬させる
				//入力層以外は前の層のsum_uをシグモイド関数に代入して出力値を求めてから次の層に伝搬させる


				if (layer_num == all_layer_siz)
				{
					//出力層の場合
					//まずsum_uをシグモイド関数に代入して出力値を求める
					layer_output[data_i][layer_num][1] = sigmoid(1.0, sum_u[data_i][layer_num][1]);
					error += pow(layer_output[data_i][layer_num][1] - data[data_i].output[0], 2) / 2;//評価関数
					diff[data_i] = layer_output[data_i][layer_num][1] - data[data_i].output[0];//誤差	
				}
				else
				{
					//出力層以外の処理

					if (layer_num == 1)
					{
						//入力層の伝搬前の処理
						//出力値の設定
						//入力層の入力はそのまま出力する(0は閾値,入力データは1-indexになっている)
						for (int now_node = 0; now_node < input_siz; now_node++)
						{
							//入力層のデータの初期化
							layer_output[data_i][layer_num][now_node] = data[data_i].input[now_node];
						}


						//layer_num+1層に各出力値を伝搬させる
						for (int now_node = 0; now_node < input_siz; now_node++)
						{

							//伝搬元のノードは閾値を含めて全て伝搬させる
							//出力層への伝搬の部分でノード1以外のノードにも伝搬させることになるが，そのノードの値を使うことはないため，この処理で問題はない
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//伝搬先のノードは閾値を含めない
								//各層の素子数分全てに伝搬させる
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}
					}
					else
					{
						//中間層の伝搬前の処理
						//sum_uをシグモイド関数に代入して出力値を求める
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							layer_output[data_i][layer_num][now_node] = sigmoid(1.0, sum_u[data_i][layer_num][now_node]);
						}

						//layer_num+1層に各出力値を伝搬させる
						for (int now_node = 0; now_node <= each_layer_siz; now_node++)
						{

							//伝搬元のノードは閾値を含めて全て伝搬させる
							//出力層への伝搬の部分でノード1以外のノードにも伝搬させることになるが，そのノードの値を使うことはないため，この処理で問題はない
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//伝搬先のノードは閾値を含めない
								//各層の素子数分全てに伝搬させる
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}

					}
				}
			}
		}



		//全てのデータについて誤差の合計を求めるのが終了した
		//****************************************************************************************************************

		//以下誤差全般を行う部分




		for (int layer_num = all_layer_siz; layer_num > 1; layer_num--)//all_layer_sizは出力層のインデックスと一致している
		{
			//全ての層につ2いて伝搬させる
			//layer_num層→layer_num-1層の伝搬について
			//出力層では次の層に伝搬させる
			//出力層以外は前の層のsum_zに現在の層の出力yに対してy(1-y)をかける前処理をしてから次の層へ伝搬させる

			if (layer_num == all_layer_siz)
			{
				//出力層の場合
				//伝搬させる
				for (int next_node = 0; next_node <= each_layer_siz; next_node++)
				{
					//伝搬させる部分
					//前の層への逆伝搬は(上の層からの入力)*(現在の層の出力)*(1-現在の層の出力)*(重み係数)

					for (int data_i = 0; data_i < data_siz; data_i++)
					{
						sum_z[data_i][layer_num - 1][next_node] += (diff[data_i]) * layer_output[data_i][layer_num][1]
							* (1 - layer_output[data_i][layer_num][1]) * weight[layer_num - 1][next_node][1];
					}
				}
				//重み係数の更新(伝播後)
				for (int next_node = 0; next_node <= each_layer_siz; next_node++)
				{
					//前の層との間の重みの更新は(上の層からの入力)*(現在の層の出力)*(1-現在の層の出力)*(前の層の出力)
					double tmp = 0;

					for (int data_i = 0; data_i < data_siz; data_i++)
					{
						//全てのデータについて合計
						tmp += (diff[data_i]) * layer_output[data_i][layer_num - 1][next_node]
							* layer_output[data_i][layer_num][1] * (1 - layer_output[data_i][layer_num][1]);
					}
					tmp /= data_siz;//平均をとって

					//重みを更新
					weight[layer_num - 1][next_node][1]
						= weight[layer_num - 1][next_node][1] - study_rate * tmp;
				}
			}
			else
			{
				//出力層以外の処理
				//ここではまず，現在の層のsum_zに現在の層の出力layer_outputに対して(1-layer_output)*layer_outputをかける前処理を行う



				if (layer_num == 2)
				{
					//入力層の1つ前の層
					//一括学習では全てのデータの誤差を合計して逆伝搬のフェーズに移るため，ここでは誤差を加算して終了
					//伝播させる部分
					//ここでは重み係数をかけて伝搬させる
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

					//重み係数の更新
					for (int now_node = 1; now_node <= each_layer_siz; now_node++)
					{
						for (int next_node = 0; next_node < input_siz; next_node++)
						{


							double tmp = 0;

							for (int data_i = 0; data_i < data_siz; data_i++)
							{
								//全てのデータについて合計
								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);
							}
							tmp /= data_siz;//平均をとって

							//重みの更新
							weight[layer_num - 1][next_node][now_node]
								= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;
						}
					}
				}
				else
				{
					//中間層


					//伝播させる部分
					//ここでは重み係数をかけて伝搬させる
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

					//重み係数の更新
					for (int now_node = 1; now_node <= each_layer_siz; now_node++)
					{
						for (int next_node = 0; next_node <= each_layer_siz; next_node++)
						{

							double tmp = 0;

							for (int data_i = 0; data_i < data_siz; data_i++)
							{
								//全てのデータについて合計
								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);
							}
							tmp /= data_siz;//平均をとって

							//重みの更新
							weight[layer_num - 1][next_node][now_node]
								= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;

						}
					}
				}
			}
		}

		//学習回数を更新を忘れない
		cnt++;

		cout << "cnt = " << cnt << " error= " << error << endl;

	} while (abs(error) > eps && cnt < rep);//誤差が閾値以下になるか学習回数の上限に達するまでは学習を継続する

	return;
}



//逐次学習
void sequential_learning(
	vector<teaching_data>& data, //教師データ
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	vector<vector<vector<double> > >& layer_output,//順方向伝搬の際の各ノードの出力
	vector<vector<vector<double> > >& rev_layer_output,//逆伝搬の際の各ノードの出力
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数
	int input_siz,//入力データの数(閾値の分も含むので入力データ+1を入力データとしている)
	int output_siz,//出力データの数
	double study_rate//学習率
) {

	//配列の添え字メモ
	//weight[i][j][k]:=i層のノードjとi+1層のノードkの間の重み係数
	//layer_out[i][j][k]:= i番目のデータに対する順方向伝搬の際のj層のノードkの出力
	//rev_layer_out[i][j][k]:= i番目のデータに対する逆方向伝搬の際のj層のノードkの出力


	//デバッグ用

	/***************************************/
	//使う変数
	int all_layer_siz = middle_layer_siz + 2;//中間層 + 入力層 + 出力層
	int data_siz = data.size();//教師データの数
	auto sum_u = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//順方向伝搬の際のシグモイド関数に入れる前の和
	auto sum_z = make_v<double>(data_siz + 10, all_layer_siz + 10, each_layer_siz + 10);//逆方向伝搬の際の出力値
	auto diff = make_v<double>(data_siz + 10);

	/********************************************/


	//乱数初期化と重みの乱数での初期化

	value_initialize(weight);//重みの初期化

	/********************************************/
	//配列は1-index

	double error = 0;//誤差(初期値は1.0)
	int cnt = 0;//学習回数

	do//誤差が閾値以下になるか学習回数の上限に達するまでは学習を継続する
	{
		error = 0.0;
		//順方向伝搬部分
		//誤差はerrorに代入する


		fill_v(sum_u, 0);//順方向出力
		fill_v(sum_z, 0);//逆方向出力
		fill_v(diff, 0);//誤差

		//***********************************************************************************************
		//以下順方向伝搬

		for (int data_i = 0; data_i < data_siz; data_i++)//データを全て入力して誤差を合計する
		{
			for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)//all_layer_sizは出力層のインデックスと一致している
			{
				//全ての層について伝搬させる//
				//layer_num層→layer_num+1層の伝搬について
				//入力層では次の層に伝搬させる
				//入力層以外は前の層のsum_uをシグモイド関数に代入して出力値を求めてから次の層に伝搬させる


				if (layer_num == all_layer_siz)
				{
					//出力層の場合
					layer_output[data_i][layer_num][1] = sigmoid(1.0, sum_u[data_i][layer_num][1]);
					
					error += pow(layer_output[data_i][layer_num][1] - data[data_i].output[0], 2) / 2;//評価関数
					diff[data_i] = layer_output[data_i][layer_num][1] - data[data_i].output[0];//誤差	

				}
				else
				{
					//出力層以外の処理

					if (layer_num == 1)
					{
						//入力層の伝搬前の処理
						//出力値の設定
						//入力層の入力はそのまま出力する(0は閾値,入力データは1-indexになっている)
						for (int now_node = 0; now_node < input_siz; now_node++)
						{
							//入力層のデータの初期化
							layer_output[data_i][layer_num][now_node] = data[data_i].input[now_node];
						}


						//layer_num+1層に各出力値を伝搬させる
						for (int now_node = 0; now_node < input_siz; now_node++)
						{

							//伝搬元のノードは閾値を含めて全て伝搬させる
							//出力層への伝搬の部分でノード1以外のノードにも伝搬させることになるが，そのノードの値を使うことはないため，この処理で問題はない
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//伝搬先のノードは閾値を含めない
								//各層の素子数分全てに伝搬させる
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}
					}
					else
					{
						//中間層の伝搬前の処理
						//sum_uをシグモイド関数に代入して出力値を求める
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							layer_output[data_i][layer_num][now_node] = sigmoid(1.0, sum_u[data_i][layer_num][now_node]);
							//cout << "layer_num " << layer_num << " now_node " << now_node << " before =" << sum_u[layer_num][now_node] << " after = " << layer_output[layer_num][now_node] << endl;
						}

						//layer_num+1層に各出力値を伝搬させる
						for (int now_node = 0; now_node <= each_layer_siz; now_node++)
						{

							//伝搬元のノードは閾値を含めて全て伝搬させる
							//出力層への伝搬の部分でノード1以外のノードにも伝搬させることになるが，そのノードの値を使うことはないため，この処理で問題はない
							for (int next_node = 1; next_node <= each_layer_siz; next_node++)
							{
								//伝搬先のノードは閾値を含めない
								//各層の素子数分全てに伝搬させる
								sum_u[data_i][layer_num + 1][next_node] += layer_output[data_i][layer_num][now_node] * weight[layer_num][now_node][next_node];
							}
						}

					}
				}
			}


			//ここで誤差逆伝搬

			for (int layer_num = all_layer_siz; layer_num > 1; layer_num--)//all_layer_sizは出力層のインデックスと一致している
			{
				//全ての層につ2いて伝搬させる
				//layer_num層→layer_num-1層の伝搬について
				//出力層では次の層に伝搬させる
				//出力層以外は前の層のsum_zに現在の層の出力yに対してy(1-y)をかける前処理をしてから次の層へ伝搬させる

				if (layer_num == all_layer_siz)
				{
					//出力層の場合
					//伝搬させる
					for (int next_node = 0; next_node <= each_layer_siz; next_node++)
					{
						//伝搬させる部分
						//前の層への逆伝搬は(上の層からの入力)*(現在の層の出力)*(1-現在の層の出力)*(重み係数)

						sum_z[data_i][layer_num - 1][next_node] += (diff[data_i]) * layer_output[data_i][layer_num][1]
							* (1 - layer_output[data_i][layer_num][1]) * weight[layer_num - 1][next_node][1];
					}

					//cout << endl;
					//重み係数の更新(伝播後)
					for (int next_node = 0; next_node <= each_layer_siz; next_node++)
					{
						//前の層との間の重みの更新は(上の層からの入力)*(現在の層の出力)*(1-現在の層の出力)*(前の層の出力)
						double tmp = 0;

						tmp += (diff[data_i]) * layer_output[data_i][layer_num - 1][next_node]
							* layer_output[data_i][layer_num][1] * (1 - layer_output[data_i][layer_num][1]);

						//重みを更新
						weight[layer_num - 1][next_node][1]
							= weight[layer_num - 1][next_node][1] - study_rate * tmp;
					}
				}
				else
				{
					//出力層以外の処理
					//ここではまず，現在の層のsum_zに現在の層の出力layer_outputに対して(1-layer_output)*layer_outputをかける前処理を行う

					if (layer_num == 2)
					{
						//入力層の1つ前の層
						//一括学習では全てのデータの誤差を合計して逆伝搬のフェーズに移るため，ここでは誤差を加算して終了
						//伝播させる部分
						//ここでは重み係数をかけて伝搬させる
						//cout << "layer_num =" << layer_num << endl;
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node < input_siz; next_node++)
							{
								sum_z[data_i][layer_num - 1][next_node] += sum_z[data_i][layer_num][now_node] * weight[layer_num - 1][next_node][now_node]
									* layer_output[data_i][layer_num][now_node] * (1 - layer_output[data_i][layer_num][now_node]);
							}
						}

						//重み係数の更新
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node < input_siz; next_node++)
							{


								double tmp = 0;

								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);

								//重みの更新
								weight[layer_num - 1][next_node][now_node]
									= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;
							}
						}
					}
					else
					{
						//中間層


						//伝播させる部分
						//ここでは重み係数をかけて伝搬させる
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node <= each_layer_siz; next_node++)
							{
								sum_z[data_i][layer_num - 1][next_node] += sum_z[data_i][layer_num][now_node] * weight[layer_num - 1][next_node][now_node]
									* layer_output[data_i][layer_num][now_node] * (1 - layer_output[data_i][layer_num][now_node]);
							}
						}

						//重み係数の更新
						for (int now_node = 1; now_node <= each_layer_siz; now_node++)
						{
							for (int next_node = 0; next_node <= each_layer_siz; next_node++)
							{

								double tmp = 0;

								//全てのデータについて合計
								tmp += sum_z[data_i][layer_num][now_node]
									* layer_output[data_i][layer_num - 1][next_node] * layer_output[data_i][layer_num][now_node]
									* (1 - layer_output[data_i][layer_num][now_node]);

								//重みの更新
								weight[layer_num - 1][next_node][now_node]
									= weight[layer_num - 1][next_node][now_node] - study_rate * tmp;

							}
						}
					}
				}

			}
		}
		//学習回数を更新を忘れない
		cnt++;

		cout << "cnt = " << cnt << " error= " << error << endl;

	} while (abs(error) > eps && cnt < rep);//誤差が閾値以下になるか学習回数の上限に達するまでは学習を継続する


	return;
}



//シグモイド関数(ゲインは1)
double sigmoid(double gain,double s) {

	return 1.0 / (1.0 + exp(-gain * s));

}



//重み係数を決定する関数
void value_initialize(vector<vector<vector<double> > >& weight) {

	// 0.0以上1.0未満の値を等確率で発生させる
	std::random_device rnd;     // 非決定的な乱数生成器を生成
	std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
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



//未学習データの値の出力
double non_teaching_data(
	vector<vector<vector<double> > >& weight,//各ノード間の重み
	teaching_data& data,//未学習データ
	int middle_layer_siz,//中間層の数
	int each_layer_siz,//各層の素子数
	int input_siz,//入力データの数(閾値の分も含むので入力データ+1を入力データとしている)
	int output_siz//出力データの数
) {

	//ここに処理

	int all_layer_siz = middle_layer_siz + 2;//中間層 + 入力層 + 出力層
	auto sum_u = make_v<double>(all_layer_siz + 10, all_layer_siz + 10);//順方向伝搬の際のシグモイド関数に入れる前の和
	fill_v(sum_u, 0);
	auto layer_output = make_v<double>(middle_layer_siz + 10, each_layer_siz + 10);//順方向伝搬の際の各ノードの出力値用
	fill_v(layer_output, 0);

	for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)
	{
		//閾値の部分の初期化
		layer_output[layer_num][0] = 1.0;
	}

	for (int layer_num = 1; layer_num <= all_layer_siz; layer_num++)//all_layer_sizは出力層のインデックスと一致している
	{
		//全ての層について伝搬させる
		//layer_num層→layer_num+1層の伝搬について
		//入力層では次の層に伝搬させる
		//入力層以外は前の層のsum_uをシグモイド関数に代入して出力値を求めてから次の層に伝搬させる


		if (layer_num == all_layer_siz)
		{
			//出力層の場合
			//まずsum_uをシグモイド関数に代入して出力値を求める
			layer_output[layer_num][1] = sigmoid(1, sum_u[layer_num][1]);
			cout << "sum_u = " << sum_u[layer_num][1] << endl;
			return layer_output[layer_num][1];

		}
		else
		{
			//出力層以外の処理

			if (layer_num == 1)
			{
				//入力層の伝搬前の処理
				//出力値の設定
				//一括学習では全てのデータの誤差を合計して逆伝搬のフェーズに移るため，ここでは誤差を加算して終了


				//入力層の入力はそのまま出力する(入力データは0-indexになっている)
				for (int now_node = 0; now_node < input_siz; now_node++)
				{
					layer_output[1][now_node] = data.input[now_node];
				}


				//layer_num+1層に各出力値を伝搬させる
				for (int now_node = 0; now_node < input_siz; now_node++)
				{

					//伝搬元のノードは閾値を含めて全て伝搬させる
					//出力層への伝搬の部分でノード1以外のノードにも伝搬させることになるが，そのノードの値を使うことはないため，この処理で問題はない
					for (int next_node = 1; next_node <= each_layer_siz; next_node++)
					{
						//伝搬先のノードは閾値を含めない
						//各層の素子数分全てに伝搬させる
						sum_u[layer_num + 1][next_node] += layer_output[layer_num][now_node] * weight[layer_num][now_node][next_node];
					}
				}
			}
			else
			{
				//中間層の伝搬前の処理
				//sum_uをシグモイド関数に代入して出力値を求める
				for (int now_node = 1; now_node <= each_layer_siz; now_node++)
				{
					layer_output[layer_num][now_node] = sigmoid(1, sum_u[layer_num][now_node]);
				}

				//layer_num+1層に各出力値を伝搬させる
				for (int now_node = 0; now_node <= each_layer_siz; now_node++)
				{

					//伝搬元のノードは閾値を含めて全て伝搬させる
					//出力層への伝搬の部分でノード1以外のノードにも伝搬させることになるが，そのノードの値を使うことはないため，この処理で問題はない
					for (int next_node = 1; next_node <= each_layer_siz; next_node++)
					{
						//伝搬先のノードは閾値を含めない
						//各層の素子数分全てに伝搬させる
						sum_u[layer_num + 1][next_node] += layer_output[layer_num][now_node] * weight[layer_num][now_node][next_node];
					}
				}

			}
		}
	}
}
