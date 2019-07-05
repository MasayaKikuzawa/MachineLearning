#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>

#include"in_out.h";
#include"neuron.h";
#include"teaching_data.h";
using namespace std;


int main() {

	//ニューロン
	
	string name;
	printf("入力データ数nを入力してください,なおここでの入力データ数は,X = [x1,x2,....xn]におけるnである: ");

	input(name);
	int input_siz = (int)string_to_double(name);

	printf("出力データ数mを入力してください,なおここでの出力データ数は,Y = [y1,y2,....ym]におけるmである: ");
	input(name);
	int output_siz = (int)string_to_double(name);


	printf("学習率を入力してください: ");
	input(name);
	double study_rate = string_to_double(name);//学習率の設定

	printf("データ数を入力してください: ");
	input(name);
	int data_siz = (int)string_to_double(name);//データ数の設定

	vector<teaching_data> data(data_siz,teaching_data(input_siz + 10,output_siz + 10));//出力データ，出力データを格納する変数を用意(念のため大きめに用意)

	/*************************/

	//ここにcsvファイルを読み込む
	ifstream ifs("data.csv");
	string str;

	if (ifs.fail())
	{
		printf("failed\n");
		exit(1);
	}

	int index = 0;//現在見ているデータのインデックス


	while (getline(ifs, str))
	{
		string tmp = "";
		istringstream stream(str);

		int cnt = 0;
		data[index].input[0] = 1;//閾値を0番目の入力値として与える
		cnt++;

		// 区切り文字がなくなるまで文字を区切っていく
		while (getline(stream, tmp, ','))
		{
			// 区切られた文字がtmpに入る
			int num = 0;
			for (int i = 0; i < tmp.size(); i++)
			{
				num = num * 10 + tmp[i] - '0';
			}
			if (cnt <= input_siz)
			{
				//入力データ
				data[index].input[cnt] = num;
			}
			else
			{
				//出力データ
				data[index].output[cnt - input_siz - 1] = num;
			}
			cnt++;
		}
		index++;
	}
	
	/***************************/


	string choice = " ";

	//学習の選択
	while (choice != "ESC") {

		print_option();
		select_choice(choice);//選択肢の選択

		if (choice == "a")
		{
			printf("\n一括学習へ\n");

			printf("中間層の層数 : ");

			input(name);
			int middle_layer_siz = (int)string_to_double(name);

			printf("各層の素子数 : ");
			input(name);
			int each_layer_siz = (int)string_to_double(name);//各層の素子数(閾値の分も考慮しているが，入力データ自体は1-indexで問題ない)

			auto weight = make_v<double>(middle_layer_siz + 10, each_layer_siz + 10, each_layer_siz + 10);//重み係数
			fill_v(weight, 0);
			//weight[i][j][k]:=i層のノードjとi+1層のノードkの間の重み係数

			auto layer_output = make_v<double>(data_siz + 10,middle_layer_siz + 10, each_layer_siz + 10);//順方向伝搬の際の各ノードの出力値用
			fill_v(layer_output, 0);

			auto rev_layer_output = make_v<double>(data_siz + 10 ,middle_layer_siz + 10, each_layer_siz + 10);//逆伝搬の際の各ノードの出力用
			fill_v(rev_layer_output, 0);

			//一括学習
			batch_learning(data, weight, layer_output, rev_layer_output, middle_layer_siz, each_layer_siz,input_siz + 1,output_siz,study_rate);
		}
		else if (choice == "b")
		{
			printf("\n逐次学習へ\n");
			
			printf("中間層の層数 : ");

			input(name);
			int middle_layer_siz = (int)string_to_double(name);

			printf("各層の素子数 : ");
			input(name);
			int each_layer_siz = (int)string_to_double(name);//各層の素子数(閾値の分も考慮しているが，入力データ自体は1-indexで問題ない)

			auto weight = make_v<double>(middle_layer_siz + 10, each_layer_siz + 10, each_layer_siz + 10);//重み係数
			fill_v(weight, 0);

			auto layer_output = make_v<double>(data_siz + 10, middle_layer_siz + 10, each_layer_siz + 10);//順方向伝搬の際の各ノードの出力値用
			fill_v(layer_output, 0);

			auto rev_layer_output = make_v<double>(data_siz + 10, middle_layer_siz + 10, each_layer_siz + 10);//逆伝搬の際の各ノードの出力用
			fill_v(rev_layer_output, 0);

			//逐次学習
			sequential_learning(data, weight, layer_output, rev_layer_output, middle_layer_siz, each_layer_siz, input_siz + 1, output_siz, study_rate);
		}
		else if (choice == "ESC")
		{
			printf("\n終了しました\n");
			return 0;
		}
		else
		{
			printf("\n選択肢を再入力してください :");
		}
	}

	return 0;
}