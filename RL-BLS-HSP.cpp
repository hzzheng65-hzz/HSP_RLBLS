#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <string.h>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace std;

// 添加全局随机数引擎
std::mt19937 gen(1);

#define ht_size 100000 //hash table size, MAXHS
#define max_lo_in_ht 500 //the max number of local optima in hash table 
#define max_weight 1000
#define min_value -2147483647
#define max_value 2147483647
#define elite_num 8

struct HashTable {
	bool* visited_solu;//访问过的解
	double visited_obj;//对应解的目标值
	bool isinHT;//0表示在哈希表中这个optima不存在，1表示在哈希表中有这个optima
	int recent_iter;//最近一次访问这个解的迭代次数
	int recent_round;//最近一次访问这个局部最优的轮次
};
struct RemovalList {
	int prev;
	int next;
};
struct EliteSol {
	bool* elite_solu;//访问过的解
	double elite_obj;//对应解的目标值
	double* elite_a;
	double* elite_b;
	int* elite_vertex;
	int* elite_address;
	int elite_hashvalue;
};
struct Score {
	int ver_id;
	double scorenum;
};

/***************读文件保存的信息***************/
int ver_num;//the total number of the graph vertex
int eg_num;//the total number of the edge vertex
int k_num;//the number of vertex inthe solution
int** edge;
double** weight;
double wmin;

/*************** Solution Information ***************/
bool* solu;
double obj;
int* vertex, * address;
bool* best_solu;
double best_obj;
int* best_vertex, * best_address;
double* a, * best_a;//the sum of weight to every vertex in the solution
double* b, * best_b;//记录各个点到除解外其他点的权重之和

/*************** Hash Table ***************/
int* rand_vec;
int hashvalue = 0;
HashTable* ht;
int cur_lo_num_in_ht;
RemovalList* rlist;
int first, last;
int best_hashvalue = 0;

/*************** Memory based Perturbation ***************/
EliteSol* elite;//记录8个局部最优解,不同的解，当找到的局部最优值大于这8个解中最差的时候插入到解中
int* flipfreq;//记录每个点反转的次数
int* elitefreq;//记录各个点在这八个局部最优解中的solu取值为1的个数
Score* score;//记录每个点的得分,在这8个局部最优解中有些点常取1或者0就有更小的概率会被翻转，同时翻转次数越小就有更大概率被反转
int cur_elite_num = 0;
int max_freq = min_value;

/*************** Local Search Array ***************/
double dmaxfortabu = 0;
int* x, * y;
int* to_out, * to_in;
int* out_list, * in_list;
int* xts1, * xts2, * yts1, * yts2;

/*************** Parameter Used ***************/
int time_limit;
double starting_time, finishing_time;// , time_limit;//开始时间
int L, L_MAX, L_MIN;//移动步长
int L_step1, L_step2, L_lastr;
int consessflag = 0;//记录连续多少次遇到相同的flag=1/2
int Iter;//迭代次数
int prev_visit;//记录是否遇到了重复解
int prev_decsent;//记录上一次遇到cycle的下降迭代次数
int lc;//上次遇到循环解的迭代次数
double wc;//用于记录连续两次下降的均值
double e;//用于确定扰动类型的参数
int maxinc;//用于确定目前遇到的最大cycle
int* H;//记录各个点最近一次0-1翻转的迭代次数
int* tabu_list;
int tabu_len;
double tabup1 = 0.1, tabup2 = 0.4;//0.1,0.4
double Q = 0.65;//0.65
double beta = 1.5;// 1.5;
double anc = 0.6, ac = 0.5;//0.6,0.5
int cyclecount = 0;
double sumdiff = 0.0;//记录当前局部最优与最优解之间的差距
int countdiff = 0;//记录diff的个数
int para;
int instance_type = 0;//1 is HSP, 2 is MDP
int perturb_num = 0;//local optima的数量

/*************** Q_learning ***************/
double** Q_table;//第一维是状态，第二维是action
double** Reward;//第一维是状态，第二维是action
int St1_Num = 5;//0:nocycle,1:10轮以内的cycle, 2:11-30轮以内的cycle,3:31-100以内的cycle，4:101及以上的cycle
int St2_Num = 4;//上一次遇到cycle 0:是10轮以内, 1:11-30轮以内,2:31-100以内，3:101及以上
int Pertype_num = 3;//扰动总数
//写一个函数判断当前状态

void ReadingMDPFile(string input_file) {
	//首先打开文件
	ifstream data(input_file);
	if (!data.is_open()) {
		cout << "Can not open the instance " << input_file << endl;
		exit(0);
	}
	//mdp
	data >> ver_num;
	if (ver_num == 2500) { data >> eg_num >> k_num; time_limit = 300; }
	else if (ver_num == 3001) { ver_num--; data >> eg_num >> k_num; time_limit = 600; }
	else if (ver_num == 5001) { ver_num--; data >> eg_num >> k_num; time_limit = 1800; }
	else { data >> k_num; eg_num = ver_num * (ver_num - 1) / 2; time_limit = 600; }
	edge = new int* [ver_num];
	weight = new double* [ver_num];
	for (int i = 0; i < ver_num; i++) {
		edge[i] = new int[ver_num];
		weight[i] = new double[ver_num];
	}
	for (int i = 0; i < ver_num; i++) {
		for (int j = 0; j < ver_num; j++) {
			edge[i][j] = 0;
			weight[i][j] = 0.0;
		}
	}
	int x1, x2;
	double x3;
	int sumedge = 0;
	for (int i = 0; i < eg_num; i++) {
		data >> x1 >> x2 >> x3;
		if (ver_num == 2500) { x1--; x2--; }
		if (x1 < 0 || x2 < 0 || x1 >= ver_num || x2 >= ver_num)
		{
			cout << "### Error of node : x1=" << x1 << ", x2=" << x2 << endl;
			exit(0);
		}
		if (x1 != x2) {
			edge[x1][x2] = 1; edge[x2][x1] = 1;
			weight[x1][x2] = x3; weight[x2][x1] = x3;
			sumedge++;
		}
	}
	eg_num = sumedge;
	cout << "the vertex num is " << ver_num << endl;
	cout << "the edge num is " << eg_num << endl;
	cout << "the ver num needed is " << k_num << endl;
}

void ReadingHSPFile(string input_file) {
	//首先打开文件
	ifstream data(input_file);
	if (!data.is_open()) {
		cout << "Can not open the instance " << input_file << endl;
		exit(0);
	}
	//文件中信息hsp
	data >> ver_num >> eg_num >> k_num;
	if (ver_num == 1000) { time_limit = 200; }
	else { time_limit = 1000; }
	cout << "the vertex num is " << ver_num << endl;
	cout << "the edge num is " << eg_num << endl;
	cout << "the ver num needed is " << k_num << endl;
	edge = new int* [ver_num];
	weight = new double* [ver_num];
	for (int i = 0; i < ver_num; i++) {
		edge[i] = new int[ver_num];
		weight[i] = new double[ver_num];
	}
	for (int i = 0; i < ver_num; i++) {
		for (int j = 0; j < ver_num; j++) {
			edge[i][j] = 0;
			weight[i][j] = 0;
		}
	}
	int x1, x2, x3;
	for (int i = 0; i < eg_num; i++) {
		data >> x1 >> x2 >> x3;
		x1--; x2--;
		if (x1 < 0 || x2 < 0 || x1 >= ver_num || x2 >= ver_num)
		{
			cout << "### Error of node : x1=" << x1 << ", x2=" << x2 << endl;
			exit(0);
		}
		if (x1 != x2) {
			edge[x1][x2] = 1; edge[x2][x1] = 1;
			weight[x1][x2] = x3; weight[x2][x1] = x3;
		}
	}
}

void BuildArray() {
	//解、目标值、两个辅助参数的初始化
	solu = new bool[ver_num];
	best_solu = new bool[ver_num];
	for (int i = 0; i < ver_num; i++) {
		solu[i] = 1;
		best_solu[i] = 1;
	}
	obj = 0.0;
	best_obj = 0.0;
	a = new double[ver_num]; best_a = new double[ver_num];
	b = new double[ver_num]; best_b = new double[ver_num];
	for (int i = 0; i < ver_num; i++) {
		a[i] = 0.0; b[i] = 0.0;
		best_a[i] = 0.0; best_b[i] = 0.0;
	}
	for (int i = 0; i < ver_num; i++) {
		for (int j = 0; j < ver_num; j++) {
			if (solu[j] == 1) {
				a[i] += weight[i][j];
			}
			else {
				b[i] += weight[i][j];
			}
		}
	}
	for (int i = 0; i < ver_num - 1; i++) {
		for (int j = i + 1; j < ver_num; j++) {
			if (solu[i] == 1 && solu[j] == 1) {
				obj += weight[i][j];
			}
		}
	}
	//点位置的初始化
	vertex = new int[ver_num];
	address = new int[ver_num];
	best_vertex = new int[ver_num];
	best_address = new int[ver_num];
	for (int i = 0; i < ver_num; i++) {
		vertex[i] = i; address[i] = i; best_vertex[i] = i; best_address[i] = i;
	}
	//哈希表的初始化
	cur_lo_num_in_ht = 0;
	ht = new HashTable[ht_size];
	for (int i = 0; i < ht_size; i++) {
		ht[i].visited_solu = new bool[ver_num];
		for (int j = 0; j < ver_num; j++) {
			ht[i].visited_solu[j] = 0;
		}
		ht[i].visited_obj = 0;
		ht[i].isinHT = 0;
		ht[i].recent_iter = 0;
		ht[i].recent_round = -1;
	}
	rlist = new RemovalList[ht_size];
	for (int i = 0; i < ht_size; i++) {
		rlist[i].next = -1;
		rlist[i].prev = -1;
	}
	first = -1;
	last = -1;
	//禁忌表和recency的初始化
	tabu_list = new int[ver_num];
	H = new int[ver_num];
	for (int i = 0; i < ver_num; i++) {
		tabu_list[i] = 0;
		H[i] = 0;
	}
	Q_table = new double* [St1_Num * St2_Num];
	Reward = new double* [St1_Num * St2_Num];
	for (int i = 0; i < St1_Num * St2_Num; i++) {
		Q_table[i] = new double[Pertype_num];
		Reward[i] = new double[Pertype_num];
		for (int j = 0; j < Pertype_num; j++) {
			Q_table[i][j] = 0;
			Reward[i][j] = 0;
		}
	}
}

void Set_Array_Memorybased() {
	elite = new EliteSol[elite_num];
	for (int i = 0; i < elite_num; i++) {
		elite[i].elite_obj = 0;
		elite[i].elite_hashvalue = 0;
		elite[i].elite_solu = new bool[ver_num];
		elite[i].elite_vertex = new int[ver_num];
		elite[i].elite_address = new int[ver_num];
		elite[i].elite_a = new double[ver_num];
		elite[i].elite_b = new double[ver_num];
		for (int j = 0; j < ver_num; j++) {
			elite[i].elite_solu[j] = 0;
			elite[i].elite_vertex[j] = 0;
			elite[i].elite_address[j] = 0;
			elite[i].elite_a[j] = 0;
			elite[i].elite_b[j] = 0;
		}
	}
	score = new Score[ver_num];
	flipfreq = new int[ver_num];
	elitefreq = new int[ver_num];
	for (int i = 0; i < ver_num; i++) {
		score[i].ver_id = i;
		score[i].scorenum = 0.0;
		flipfreq[i] = 0;
		elitefreq[i] = 0;
	}
	//把局部最优解保存到elite中，更新cur_elite_num,记录worst_obj
}

void Set_Array_LS() {
	//设置dmax
	for (int i = 0; i < ver_num - 1; i++) {
		for (int j = i + 1; j < ver_num; j++) {
			if (weight[i][j] > dmaxfortabu) { dmaxfortabu = weight[i][j]; }
		}
	}
	//localsearch
	x = new int[ver_num]; y = new int[ver_num];
	to_out = new int[ver_num]; to_in = new int[ver_num];
	for (int i = 0; i < ver_num; i++) {
		to_out[i] = -1; to_in[i] = -1;
		x[i] = -1; y[i] = -1;
	}
	//twoswap
	int size_num = ver_num * ver_num;
	xts1 = new int[size_num]; yts1 = new int[size_num];
	xts2 = new int[size_num]; yts2 = new int[size_num];
	for (int i = 0; i < size_num; i++) {
		xts1[i] = -1; xts2[i] = -1; yts1[i] = -1; yts2[i] = -1;
	}
	//mem\randperturb
	out_list = new int[ver_num];
	in_list = new int[ver_num];
	for (int i = 0; i < ver_num; i++) {
		out_list[i] = -1;
		in_list[i] = -1;
	}
}

bool compare(const EliteSol& a, const EliteSol& b) {
	return a.elite_obj > b.elite_obj;//降序排列
}

bool comparescore(const Score& a, const Score& b) {
	return a.scorenum > b.scorenum;//降序排列
}

void Save_to_EliteSol(int i) {
	elite[i].elite_obj = obj;
	elite[i].elite_hashvalue = hashvalue;
	for (int j = 0; j < ver_num; j++) {
		elite[i].elite_solu[j] = solu[j];
		elite[i].elite_vertex[j] = vertex[j];
		elite[i].elite_address[j] = address[j];
		elite[i].elite_a[j] = a[j];
		elite[i].elite_b[j] = b[j];
	}
}

void Save_bestsolu() {
	for (int i = 0; i < ver_num; i++) {
		best_solu[i] = solu[i];
		best_a[i] = a[i]; best_b[i] = b[i];
		best_vertex[i] = vertex[i];
		best_address[i] = address[i];
	}
	best_obj = obj; best_hashvalue = hashvalue;
	cout << best_obj << ", " << (finishing_time - starting_time) / CLOCKS_PER_SEC << endl;
}

void update_para(int out_ver, int in_ver) {
	if (in_ver > ver_num) {//delete
		obj = obj - a[out_ver];
		for (int i = 0; i < ver_num; i++) {
			a[i] = a[i] - weight[i][out_ver];
			b[i] = b[i] + weight[i][out_ver];
		}
	}
	else if (out_ver > ver_num) {//add
		obj = obj + a[in_ver];
		for (int i = 0; i < ver_num; i++) {
			a[i] = a[i] + weight[i][in_ver];
			b[i] = b[i] - weight[i][in_ver];
		}
	}
	else {//swap
		obj = obj + a[in_ver] - a[out_ver] - weight[out_ver][in_ver];
		for (int i = 0; i < ver_num; i++) {
			a[i] = a[i] - weight[i][out_ver] + weight[i][in_ver];
			b[i] = b[i] + weight[i][out_ver] - weight[i][in_ver];
		}
	}
}

void SetVector() {
	rand_vec = new int[ver_num];
	std::uniform_int_distribution<> dis(0, 131072);
	for (int i = 0; i < ver_num; i++) {
		rand_vec[i] = dis(gen);
	}
}

void RandSort(int* list, int len) {
	int x, y, z;
	std::uniform_int_distribution<> dis(0, len - 1);
	y = dis(gen);
	z = dis(gen);
	for (int i = 0; i < len; i++) {
		x = list[y];
		list[y] = list[z];
		list[z] = x;
	}
}

void Initialize() {
	for (int r_num = 0; r_num < (ver_num - k_num); r_num++) {
		int* outlist;
		outlist = new int[ver_num];
		for (int i = 0; i < ver_num; i++) {
			outlist[i] = i;
		}
		int outnum = 0;
		double min_weight = max_value;// 1000 * ver_num * (ver_num - 1);
		int out_ver = -1;
		for (int i = 0; i < ver_num; i++) {
			if (solu[i] == 1) {
				if (a[i] < min_weight) {
					outnum = 0;
					min_weight = a[i];
					outlist[outnum] = i;
					outnum++;
				}
				else if (fabs(a[i] - min_weight) < 1e-6) {
					outlist[outnum] = i;
					outnum++;
				}
			}
		}
		std::uniform_int_distribution<> dis(0, outnum - 1);
		out_ver = outlist[dis(gen)];
		update_para(out_ver, ver_num + 1);
		solu[out_ver] = 0;
	}
	finishing_time = (double)clock();
	cout.setf(ios::fixed);
	cout << std::setprecision(6) << "obj(initial solution) = " << obj << endl;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 1) {
			hashvalue += rand_vec[i];
		}
	}
	//确定点的位置
	int num = 0, cnum = k_num;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 1) { vertex[num] = i; address[i] = num; num++; }
		if (solu[i] == 0) { vertex[cnum] = i; address[i] = cnum; cnum++; }
	}
	Save_bestsolu();
}

void Move(int out_ver, int in_ver) {
	update_para(out_ver, in_ver);
	solu[out_ver] = 0;
	solu[in_ver] = 1;
	H[out_ver] = Iter;
	H[in_ver] = Iter;
	std::uniform_int_distribution<> dis_len(0, static_cast<int>(ceil(tabup2 * (k_num))) - 1);
	tabu_len = ceil(tabup1 * k_num) + dis_len(gen);
	tabu_list[out_ver] = Iter + tabu_len;
	tabu_list[in_ver] = Iter + ceil(0.7 * tabu_len);
	Iter++;
	hashvalue = hashvalue - rand_vec[out_ver] + rand_vec[in_ver];
	//更新vertex和address
	int x = address[out_ver];
	vertex[x] = in_ver; vertex[address[in_ver]] = out_ver;
	address[out_ver] = address[in_ver];
	address[in_ver] = x;
	//更新flipfreq
	flipfreq[out_ver]++; flipfreq[in_ver]++;
	if (flipfreq[out_ver] > max_freq) { max_freq = flipfreq[out_ver]; }
	if (flipfreq[in_ver] > max_freq) { max_freq = flipfreq[in_ver]; }
}

void insert_to_list(int& first, int& last, int cur_index) {
	if (first == -1) {
		last = cur_index;
		first = cur_index;
		rlist[cur_index].next = -1;
		rlist[cur_index].prev = -1;
	}
	else {
		rlist[cur_index].next = first;
		rlist[cur_index].prev = -1;
		rlist[first].prev = cur_index;
		first = cur_index;
	}
}

void delete_from_list(int& first, int& last, int cur_index) {
	if (rlist[cur_index].prev == -1) {
		first = rlist[cur_index].next;
	}
	else {
		rlist[rlist[cur_index].prev].next = rlist[cur_index].next;
	}
	if (rlist[cur_index].next == -1) {
		last = rlist[cur_index].prev;
		rlist[rlist[cur_index].prev].next = -1;
	}
	else {
		rlist[rlist[cur_index].next].prev = rlist[cur_index].prev;
	}
	rlist[cur_index].next = -1;
	rlist[cur_index].prev = -1;
}

int prev_round = 0;
int prev_cycle = 0;
int last_cycle = 0;
int PreviousEncounter() {
	int cur_index = 0;//此次局部搜索获得的最优解在哈希表中的键值
	cur_index = hashvalue % (ht_size);
	int flag = 0;
	while (true) {
		if (ht[cur_index].isinHT == 0) {//这个解的键值不在哈希表中,即这个局部最优没有在之前搜索过
			cur_lo_num_in_ht++;//哈希表中保存的最优解增加1
			ht[cur_index].isinHT = 1;
			ht[cur_index].recent_iter = Iter;
			ht[cur_index].recent_round = perturb_num;
			ht[cur_index].visited_obj = obj;
			for (int i = 0; i < ver_num; i++) {
				ht[cur_index].visited_solu[i] = solu[i];
			}
			insert_to_list(first, last, cur_index);
			flag = -1;
			/******************** Elite Sol******************************/
			if (cur_elite_num < elite_num) {
				Save_to_EliteSol(cur_elite_num);
				sort(elite, elite + elite_num, compare);
				cur_elite_num++;
				for (int i = 0; i < ver_num; i++) {
					elitefreq[i] += solu[i];
				}
			}
			else {
				if (obj > elite[elite_num - 1].elite_obj) {
					for (int i = 0; i < ver_num; i++) {
						elitefreq[i] += solu[i];
						elitefreq[i] -= elite[elite_num - 1].elite_solu[i];
					}
					Save_to_EliteSol(elite_num - 1);
					sort(elite, elite + elite_num, compare);
				}
			}
			/******************** Elite Sol******************************/
			break;
		}
		else {//这个解的键值在哈希表中，对比两个解是否一样
			bool identical = 1;
			if (fabs(obj - ht[cur_index].visited_obj) > 1e-6) {//对比目标值是否一样
				cur_index = (cur_index + 1) % (ht_size);// + 1
			}
			else {
				for (int i = 0; i < ver_num; i++) {
					if (ht[cur_index].visited_solu[i] != solu[i]) {
						cur_index = (cur_index + 1) % (ht_size);// + 1
						identical = 0;
						break;
					}
				}
				if (identical != 0) {
					flag = ht[cur_index].recent_iter;
					prev_round = ht[cur_index].recent_round;//new
					ht[cur_index].recent_iter = Iter;
					ht[cur_index].recent_round = perturb_num;
					delete_from_list(first, last, cur_index);
					insert_to_list(first, last, cur_index);
					break;
				}
			}
		}
	}
	if (cur_lo_num_in_ht > max_lo_in_ht) {
		ht[last].isinHT = 0;
		ht[last].recent_iter = 0;
		ht[last].recent_round = -1;
		delete_from_list(first, last, last);
		cur_lo_num_in_ht--;
	}
	return flag;
}

void DetermineJumpMagnitude() {
	L_lastr = prev_visit;
	prev_visit = PreviousEncounter();
	int step1 = 1;
	if (L_lastr == -1 && prev_visit == -1) {//上一轮和这一轮都遇到了新解
		consessflag++; step1 = L_step1 + consessflag;
	}
	else if (L_lastr != -1 && prev_visit != -1) {
		consessflag++; step1 = L_step1 + consessflag;
	}
	else { consessflag = 0; }
	//step1 = 1;/*********************************************************************/
	if (prev_visit != -1) {//遇到了重复的局部最优
		cyclecount++;
		wc = ((double)prev_decsent + (double)Iter - (double)prev_visit) / (double)2;
		prev_decsent = Iter - lc;
		lc = Iter;
		L += step1;
	}
	else if ((double)Iter - (double)lc > ceil(beta * wc)) {
		L -= step1;
	}
	if (L > L_MAX) {
		L = L_MAX;
	}
	else if (L < L_MIN) {
		L = L_MIN;
	}
}

void DirectPerturbationattr2dmax() {
	double dminout = max_value, dmaxin = min_value;
	int verout = -1, verin = -1;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 1) {
			if (dminout > a[i]) {
				dminout = a[i]; verout = i;
			}
		}
		if (solu[i] == 0) {
			if (dmaxin < a[i]) {
				dmaxin = a[i]; verin = i;
			}
		}
	}
	int xnum = 0, ynum = 0;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 1) {
			if (a[i] <= dminout + dmaxfortabu * 2.0) {
				x[xnum] = i; xnum++;
			}
		}
		if (solu[i] == 0) {
			if (a[i] >= dmaxin - dmaxfortabu * 2.0) {
				y[ynum] = i; ynum++;
			}
		}
	}
	double improve = 0;
	int in_ver = -1, out_ver = -1;
	double max = min_value;
	int num_to_out = 0, num_to_in = 0;
	for (int i = 0; i < xnum; i++) {
		for (int j = 0; j < ynum; j++) {
			improve = a[y[j]] - a[x[i]] - weight[x[i]][y[j]];
			if (tabu_list[y[j]] < Iter && tabu_list[x[i]] < Iter) {
				if (improve > max) {
					max = improve;
					num_to_in = 0; num_to_out = 0;
					to_in[num_to_in] = y[j];
					to_out[num_to_out] = x[i];
					num_to_in++; num_to_out++;
				}
				else if (fabs(improve - max) <= 1e-6) {
					to_in[num_to_in] = y[j];
					to_out[num_to_out] = x[i];
					num_to_in++; num_to_out++;
				}
			}
			else {
				if (obj + improve > best_obj) {
					if (improve > max) {
						max = improve;
						num_to_in = 0; num_to_out = 0;
						to_in[num_to_in] = y[j];
						to_out[num_to_out] = x[i];
						num_to_in++; num_to_out++;
					}
					else if (fabs(improve - max) <= 1e-6) {
						to_in[num_to_in] = y[j];
						to_out[num_to_out] = x[i];
						num_to_in++; num_to_out++;
					}
				}
			}
		}
	}
	if (num_to_in != 0) {
		std::uniform_int_distribution<int> dist(0, num_to_in - 1);
		int id = dist(gen);
		out_ver = to_out[id];
		in_ver = to_in[id];
	}
	if (in_ver != -1 && out_ver != -1) {
		Move(out_ver, in_ver);
		Iter--;
	}
	Iter++;
}

void RandomPerturbation(int L) {
	//找L个移出的点和L个移入的点
	int x1 = 0, x2 = 0;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 1) {
			out_list[x1] = i;
			x1++;
		}
		else {
			in_list[x2] = i;
			x2++;
		}
	}
	RandSort(out_list, k_num);
	RandSort(in_list, ver_num - k_num);
	int y1, y2;
	for (int i = 0; i < L; i++) {
		y1 = out_list[i];
		y2 = in_list[i];
		Move(y1, y2);
	}
}

void MemoryPerturbation(int L) {
	//首先计算每个点的score
	for (int i = 0; i < ver_num; i++) {
		int ver = score[i].ver_id;
		score[i].scorenum = (double)elitefreq[ver] * ((double)cur_elite_num - (double)elitefreq[ver]);
		score[i].scorenum /= ((double)cur_elite_num * (double)cur_elite_num);
		score[i].scorenum += (0.3 * ((double)1 - (double)flipfreq[ver] / (double)max_freq));
	}
	sort(score, score + ver_num, comparescore);
	int out_num = 0, in_num = 0;
	for (int i = 0; i < ver_num; i++) {
		if (solu[score[i].ver_id] == 1) {
			out_list[out_num] = score[i].ver_id;
			out_num++;
		}
		if (solu[score[i].ver_id] == 0) {
			in_list[in_num] = score[i].ver_id;
			in_num++;
		}
	}
	//扰动
	for (int i = 0; i < L; i++) {
		Move(out_list[i], in_list[i]);
	}
}

void General_Move(int out_ver, int in_ver) {
	solu[out_ver] = 0;
	solu[in_ver] = 1;
	H[out_ver] = Iter;
	H[in_ver] = Iter;
	std::uniform_int_distribution<> dis_len(0, static_cast<int>(ceil(tabup2 * (k_num))) - 1);
	tabu_len = ceil(tabup1 * k_num) + dis_len(gen);
	tabu_list[out_ver] = Iter + tabu_len;
	tabu_list[in_ver] = Iter + ceil(0.7 * tabu_len);
	Iter++;
	hashvalue = hashvalue - rand_vec[out_ver] + rand_vec[in_ver];
	//更新vertex和address
	int x = address[out_ver];
	vertex[x] = in_ver; vertex[address[in_ver]] = out_ver;
	address[out_ver] = address[in_ver];
	address[in_ver] = x;
	//更新flipfreq
	flipfreq[out_ver]++; flipfreq[in_ver]++;
	if (flipfreq[out_ver] > max_freq) { max_freq = flipfreq[out_ver]; }
	if (flipfreq[in_ver] > max_freq) { max_freq = flipfreq[in_ver]; }
}

double prob_para1 = 1, prob_para2 = 1;
void GernalSwap(int curL) {
	//cout << "curL = " << curL << endl;
	vector<bool> used(ver_num, false);  // 防止重复选入
	int out_num = 0, in_num = 0;
	// Step 1: 选出 curL 个最差的点
	for (int i = 0; i < curL; ++i) {
		double min_a = max_value;
		int outver = -1;
		for (int j = 0; j < ver_num; ++j) {
			std::uniform_real_distribution<> dis(0.0, 1.0);
			double prob = dis(gen);
			if (solu[j] == 1 && !used[j] && a[j] < min_a && prob <= prob_para1) {
				min_a = a[j]; outver = j;
			}
		}
		if (outver != -1) {
			out_list[out_num] = outver; out_num++;
			used[outver] = true;
			update_para(outver, ver_num + 1);
		}
	}

	// Step 2: 选出 curL 个最好的未选点
	for (int i = 0; i < curL; ++i) {
		double max_a = min_value;
		int inver = -1;
		for (int j = 0; j < ver_num; ++j) {
			std::uniform_real_distribution<> dis(0.0, 1.0);
			double prob = dis(gen);
			if (solu[j] == 0 && !used[j] && a[j] > max_a && prob <= prob_para2) {
				max_a = a[j]; inver = j;
			}
		}
		if (inver != -1) {
			in_list[in_num] = inver; in_num++;
			used[inver] = true;
			update_para(ver_num + 1, inver);
		}
	}

	// Step 3: Swap using General_Move (which will update everything)
	int num_swap = std::min(out_num, in_num);
	for (int i = 0; i < num_swap; ++i) {
		General_Move(out_list[i], in_list[i]);
	}
}

double reward1, reward2, reward_para;
double reward_discount = 0.95; //用于衰减历史奖励信息
void Compute_Reward(int prevst0, int prevper, int curst1, int curst2) {
	double immediate_reward = 0.0;

	if (prev_visit == -1) {
		immediate_reward += reward1;//新解强奖励
		immediate_reward += (curst2 - reward2);
		if (prevper == 0) { immediate_reward += 4; }
		else if (prevper == 1) { immediate_reward += 3; }
		else if (prevper == 2) { immediate_reward += 1; }
		else if (prevper == 3) { immediate_reward += 2; }
	}
	else {
		if (curst1 == 1) { immediate_reward -= 2; }
		else if (curst1 == 2) { immediate_reward -= 3; }
		else if (curst1 == 3) { immediate_reward -= 4; }
		else if (curst1 == 4) { immediate_reward -= 5; }
		immediate_reward += (curst2 - reward2);
	}

	Reward[prevst0][prevper] = reward_discount * Reward[prevst0][prevper] + immediate_reward; 
}

int prev_st0 = 0;
int prev_per = -1;
double prev_lo;
double sigma = 0.5;//accept the best Q 
double alpha = 0.1;//leraning rate
double gamma1 = 0.9;//discount rate
int st1_frame1 = 5, st1_frame2 = 10, st1_frame3 = 20, st1_frame4, st1_frame5;
int st2_frame1 = 5, st2_frame2 = 10, st2_frame3 = 20, st2_frame4, st2_frame5;
int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
void SetState1(int& cur_st1) {
	if (St1_Num == 6) {
		st1_frame1 = 1, st1_frame2 = 5, st1_frame3 = 15, st1_frame4 = 50;
		if (prev_visit == -1) { cur_st1 = 0; }
		else if (perturb_num - prev_round <= st1_frame1) { cur_st1 = 1; }
		else if (perturb_num - prev_round <= st1_frame2) { cur_st1 = 2; }
		else if (perturb_num - prev_round <= st1_frame3) { cur_st1 = 3; }
		else if (perturb_num - prev_round <= st1_frame4) { cur_st1 = 4; }
		else if (perturb_num - prev_round > st1_frame4) { cur_st1 = 5; }
	}
	else {
		if (prev_visit == -1) { cur_st1 = 0; }
		else if (perturb_num - prev_round <= st1_frame1) { cur_st1 = 1; }
		else if (perturb_num - prev_round <= st1_frame2) { cur_st1 = 2; }
		else if (perturb_num - prev_round <= st1_frame3) { cur_st1 = 3; }
		else if (perturb_num - prev_round > st1_frame3) { cur_st1 = 4; }
	}
}

void SetState2(int& cur_st2) {
	if (St2_Num == 5) {
		st2_frame1 = 2, st2_frame2 = 6, st2_frame3 = 15, st2_frame4 = 30;
		if (perturb_num - last_cycle <= st2_frame1) { cur_st2 = 0; }
		else if (perturb_num - last_cycle <= st2_frame2) { cur_st2 = 1; }
		else if (perturb_num - last_cycle <= st2_frame3) { cur_st2 = 2; }
		else if (perturb_num - last_cycle <= st2_frame4) { cur_st2 = 3; }
		else if (perturb_num - last_cycle > st2_frame4) { cur_st2 = 4; }
	}
	else {
		if (perturb_num - last_cycle <= st2_frame1) { cur_st2 = 0; }
		else if (perturb_num - last_cycle <= st2_frame2) { cur_st2 = 1; }
		else if (perturb_num - last_cycle <= st2_frame3) { cur_st2 = 2; }
		else if (perturb_num - last_cycle > st2_frame3) { cur_st2 = 3; }
	}
}

// 单次 run 的统计量
vector<vector<vector<long long>>> state_action_cnt;   // [st1][st2][a]
vector<vector<long long>> state_visit_cnt;            // [st1][st2]
void InitPolicyStats() {
	state_action_cnt.assign(
		St1_Num,
		vector<vector<long long>>(St2_Num, vector<long long>(Pertype_num, 0))
	);

	state_visit_cnt.assign(
		St1_Num,
		vector<long long>(St2_Num, 0)
	);
}

void DeterminePerturbationType() {
	sigma = 0.5 + 0.35 * tanh((perturb_num - 5000) / 2000.0);
	int cur_st1 = -1, cur_st2 = -1;
	if (perturb_num > 1) {
		//确定state1
		SetState1(cur_st1);
		//确定state2
		last_cycle = prev_cycle;
		if (prev_visit != -1) { prev_cycle = perturb_num; }
		SetState2(cur_st2);
		// 保存上一轮状态-动作，并确定执行该动作后到达的新状态
		const int old_st0 = prev_st0;
		const int old_per = prev_per;
		const int next_st0 = cur_st1 * St2_Num + cur_st2;

		// 按论文形式更新上一轮状态-动作对应的 Reward 矩阵元素
		Compute_Reward(old_st0, old_per, cur_st1, cur_st2);

		// 计算新状态下的最大 Q 值
		double nextmaxQ = Q_table[next_st0][0];
		for (int i = 1; i < Pertype_num; i++) {
			if (nextmaxQ < Q_table[next_st0][i]) {
				nextmaxQ = Q_table[next_st0][i];
			}
		}

		// 更新上一轮的 Q(old_st0, old_per)
		Q_table[old_st0][old_per] = (1 - alpha) * Q_table[old_st0][old_per]
			+ alpha * (Reward[old_st0][old_per] + gamma1 * nextmaxQ);

		// 将新状态保存为下一轮的上一状态
		prev_st0 = next_st0;
	}
	//然后选择sigma的概率选择Q(st0, action)最大的扰动，并保存在cur_per里面
	std::uniform_real_distribution<> dis(0.0, 1.0);
	double prob = dis(gen);
	if (prob <= sigma) {
		//选Q最大的
		double nextmaxQ = 0;
		for (int i = 0; i < Pertype_num; i++) {
			if (i == 0) { nextmaxQ = Q_table[prev_st0][i]; prev_per = i; }
			else {
				if (nextmaxQ < Q_table[prev_st0][i]) { nextmaxQ = Q_table[prev_st0][i]; prev_per = i; }
				else if (fabs(nextmaxQ - Q_table[prev_st0][i]) <= 1e-6) {
					std::uniform_real_distribution<> dis(0.0, 1.0);
					double prob1 = dis(gen);
					if (prob1 <= 0.5) { prev_per = i; }
				}
			}
		}
	}
	else {
		//随机选
		std::uniform_int_distribution<> dis_per(0, Pertype_num - 1);
		prev_per = dis_per(gen);
	}
	int curL = L;
	if (prev_per == 0) {
		sum0++;
		for (int i = 0; i < curL; i++) {
			DirectPerturbationattr2dmax();
			if (obj > best_obj) {
				finishing_time = (double)clock();
				Save_bestsolu();
			}
		}
	}
	else if (prev_per == 1) { sum1++; MemoryPerturbation(curL); }
	else if (prev_per == 2) { sum2++; RandomPerturbation(curL); }
	else if (prev_per == 3) { sum3++; GernalSwap(curL); }
	//cout << sum0 << "\t" << sum1 << "\t" << sum2 << "\t" << sum3 << "\t" << cur_st1 << "\t" << cur_st2 << endl;
	// ===== 新增：记录当前状态下所选动作 =====
	if (perturb_num > 1) {
		state_action_cnt[cur_st1][cur_st2][prev_per]++;
		state_visit_cnt[cur_st1][cur_st2]++;
	}
}

/************************************************* K CHANGE START ************************************************/
void ini_K_Local_Search() {
	//确定X和Y
	double cur_obj;
	do {
		if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
		int dminout = max_value, dmaxin = min_value;
		int verout = -1, verin = -1;
		for (int i = 0; i < ver_num; i++) {
			if (solu[i] == 1) {
				if (dminout > a[i]) {
					dminout = a[i]; verout = i;
				}
			}
			if (solu[i] == 0) {
				if (dmaxin < a[i]) {
					dmaxin = a[i]; verin = i;
				}
			}
		}

		int xnum = 0, ynum = 0;
		for (int i = 0; i < ver_num; i++) {
			if (solu[i] == 1) {
				if (a[i] <= dminout + weight[verout][verin]) {
					x[xnum] = i; xnum++;
				}
			}
			if (solu[i] == 0) {
				if (a[i] >= dmaxin - weight[verout][verin]) {
					y[ynum] = i; ynum++;
				}
			}
		}
		cur_obj = obj;
		int improve = 0;
		int in_ver = -1, out_ver = -1;
		int max = min_value;
		for (int i = 0; i < xnum; i++) {
			for (int j = 0; j < ynum; j++) {
				improve = a[y[j]] - a[x[i]] - weight[x[i]][y[j]];
				if (improve > max) {
					max = improve;
					in_ver = y[j];
					out_ver = x[i];
				}
			}
		}
		if (in_ver != -1 && out_ver != -1 && max > 0) {
			Move(out_ver, in_ver);
		}
	} while (obj > cur_obj);
}

void ini_K_twoswap() {
	double cur_obj;
	do {
		if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
		int maxsum = min_value, minsum = max_value;
		int vera = -1, verb = -1, verc = -1, verd = -1;
		//在可以移入的点中找两个计算出来ingain最大的点
		for (int i = 0; i < ver_num - 1; i++) {
			if (solu[i] == 0) {
				for (int j = i + 1; j < ver_num; j++) {
					if (solu[j] == 0) {
						int ingain = a[i] + a[j] + weight[i][j];
						if (ingain > maxsum) {
							maxsum = ingain; verc = j; verd = i;
						}
					}
				}
			}
			if (solu[i] == 1) {
				for (int j = i + 1; j < ver_num; j++) {
					if (solu[j] == 1) {
						int outgain = a[i] + a[j] - weight[i][j];
						if (outgain < minsum) {
							minsum = outgain; vera = j; verb = i;
						}
					}
				}
			}
		}
		int xnum = 0;
		xts1[xnum] = vera;
		xts2[xnum] = verb; xnum++;
		int ynum = 0;
		yts1[ynum] = verc;
		yts2[ynum] = verd; ynum++;
		cur_obj = obj;
		int improve = 0;
		int in_ver1 = 0, out_ver1 = 0;
		int in_ver2 = 0, out_ver2 = 0;
		int max = min_value;
		for (int i = 0; i < xnum; i++) {
			if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
			for (int j = 0; j < ynum; j++) {
				if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
				improve = a[yts1[j]] + a[yts2[j]] - a[xts1[i]] - a[xts2[i]] + weight[yts1[j]][yts2[j]] + weight[xts1[i]][xts2[i]];
				improve = improve - weight[xts1[i]][yts1[j]] - weight[xts1[i]][yts2[j]];
				improve = improve - weight[xts2[i]][yts1[j]] - weight[xts2[i]][yts2[j]];
				/**/if (improve > max) {
					max = improve;
					in_ver1 = yts1[j]; out_ver1 = xts1[i];
					in_ver2 = yts2[j]; out_ver2 = xts2[i];
				}
			}
		}
		if (max > 0) {
			Move(out_ver2, in_ver2);
			Iter--;
			Move(out_ver1, in_ver1);
		}
	} while (obj > cur_obj);
}

void full_K_Local_Search() {
	double cur_obj;
	do {
		if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }

		cur_obj = obj;
		double dminout = max_value;
		double dmaxin = min_value;
		int verout = -1;
		int verin = -1;
		// ==============================
		// 1. Find MinToOut and MaxToIn
		// ==============================
		for (int i = 0; i < ver_num; i++) {
			if (solu[i] == 1) {
				if (dminout > a[i]) {
					dminout = a[i]; verout = i;
				}
			}
			if (solu[i] == 0) {
				if (dmaxin < a[i]) {
					dmaxin = a[i]; verin = i;
				}
			}
		}

		// ==============================
		// 3. Construct X and Y
		// X = {i in U | alpha_i <= MinToOut + beta}
		// Y = {j notin U | alpha_j >= MaxToIn - beta}
		// ==============================
		double beta = weight[verout][verin] - wmin;
		int xnum = 0; int ynum = 0;
		for (int i = 0; i < ver_num; i++) {
			if (solu[i] == 1) {
				if (a[i] <= dminout + beta) {
					x[xnum] = i; xnum++;
				}
			}
			if (solu[i] == 0) {
				if (a[i] >= dmaxin - beta) {
					y[ynum] = i; ynum++;
				}
			}
		}
		// ==============================
		// 4. Find the best Swap move in X x Y
		// ==============================
		double best_gain = min_value;
		int in_ver = -1;
		int out_ver = -1;
		for (int i = 0; i < xnum; i++) {
			for (int j = 0; j < ynum; j++) {
				if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
				double gain = a[y[j]] - a[x[i]] - weight[x[i]][y[j]];
				if (gain > best_gain) {
					best_gain = gain;
					in_ver = y[j];
					out_ver = x[i];
				}
			}
		}
		// ==============================
		// 5. Perform the best improving move
		// ==============================
		if (in_ver != -1 && out_ver != -1 && best_gain > 0) {
			Move(out_ver, in_ver);
		}

	} while (obj > cur_obj + 1e-9);
}

void full_K_twoswap() {
	double cur_obj;

	do {
		if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) {
			break;
		}

		// ==============================
		// 1. Find MinToOutP and MaxToInP
		// ==============================
		double MinToOutP = max_value;
		double MaxToInP = min_value;

		int ref_out1 = -1, ref_out2 = -1;
		int ref_in1 = -1, ref_in2 = -1;

		// selected pair: eta_ab = alpha_a + alpha_b - w_ab
		for (int i = 0; i < ver_num - 1; i++) {
			if (solu[i] == 1) {
				for (int j = i + 1; j < ver_num; j++) {
					if (solu[j] == 1) {
						double eta = a[i] + a[j] - weight[i][j];

						if (eta < MinToOutP) {
							MinToOutP = eta;
							ref_out1 = i;
							ref_out2 = j;
						}
					}
				}
			}
		}

		// unselected pair: theta_cd = alpha_c + alpha_d + w_cd
		for (int i = 0; i < ver_num - 1; i++) {
			if (solu[i] == 0) {
				for (int j = i + 1; j < ver_num; j++) {
					if (solu[j] == 0) {
						double theta = a[i] + a[j] + weight[i][j];

						if (theta > MaxToInP) {
							MaxToInP = theta;
							ref_in1 = i;
							ref_in2 = j;
						}
					}
				}
			}
		}

		// ==============================
		// 2. Compute lambda
		// lambda = delta_ref - 4 * wmin
		// ==============================
		double delta_ref =
			weight[ref_out1][ref_in1] +
			weight[ref_out1][ref_in2] +
			weight[ref_out2][ref_in1] +
			weight[ref_out2][ref_in2];

		double lambda = delta_ref - 4.0 * wmin;
		double para_test = 1.0;//0.5***************************************************************************
			// ==============================
			// 3. Construct XP and YP
			// XP = {(a,b) | eta_ab <= MinToOutP + lambda}
			// YP = {(c,d) | theta_cd >= MaxToInP - lambda}
			// ==============================
		int xnum = 0;
		int ynum = 0;

		for (int i = 0; i < ver_num - 1; i++) {
			if (solu[i] == 1) {
				for (int j = i + 1; j < ver_num; j++) {
					if (solu[j] == 1) {
						double eta = a[i] + a[j] - weight[i][j];

						if (eta <= MinToOutP + para_test * lambda) {
							xts1[xnum] = i;
							xts2[xnum] = j;
							xnum++;
						}
					}
				}
			}
		}

		for (int i = 0; i < ver_num - 1; i++) {
			if (solu[i] == 0) {
				for (int j = i + 1; j < ver_num; j++) {
					if (solu[j] == 0) {
						double theta = a[i] + a[j] + weight[i][j];

						if (theta >= MaxToInP - para_test * lambda) {
							yts1[ynum] = i;
							yts2[ynum] = j;
							ynum++;
						}
					}
				}
			}
		}


		// ==============================
		// 4. Search the best SwapPair move in XP × YP
		// ==============================
		cur_obj = obj;

		double best_gain = min_value;

		int best_out1 = -1, best_out2 = -1;
		int best_in1 = -1, best_in2 = -1;

		for (int i = 0; i < xnum; i++) {
			if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) {
				break;
			}

			int out1 = xts1[i];
			int out2 = xts2[i];

			for (int j = 0; j < ynum; j++) {
				if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) {
					break;
				}

				int in1 = yts1[j];
				int in2 = yts2[j];

				double eta = a[out1] + a[out2] - weight[out1][out2];
				double theta = a[in1] + a[in2] + weight[in1][in2];

				double delta =
					weight[out1][in1] +
					weight[out1][in2] +
					weight[out2][in1] +
					weight[out2][in2];

				double gain = theta - eta - delta;

				if (gain > best_gain) {
					best_gain = gain;

					best_out1 = out1;
					best_out2 = out2;
					best_in1 = in1;
					best_in2 = in2;
				}
			}
		}

		// ==============================
		// 5. Perform the best improving SwapPair move
		// ==============================
		if (best_out1 != -1 && best_out2 != -1 &&
			best_in1 != -1 && best_in2 != -1 &&
			best_gain > 0) {

			Move(best_out2, best_in2);
			Iter--;
			Move(best_out1, best_in1);
		}

	} while (obj > cur_obj + 1e-9);
}

void K_Delete() {
	//delete one with minimum contribution
	double min_contribution = max_value;
	int ver_out = -1;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 1 && a[i] < min_contribution) {
			min_contribution = a[i]; ver_out = i;
		}
	}
	solu[ver_out] = 0;//update
	int x = address[ver_out];
	int in_ver = vertex[k_num - 1];
	vertex[x] = in_ver; vertex[address[in_ver]] = ver_out;
	address[ver_out] = address[in_ver];
	address[in_ver] = x;
	update_para(ver_out, ver_num + 1);
	hashvalue = hashvalue - rand_vec[ver_out];
}

void K_Add() {
	//add one with the maximum contribution
	double max_contribution = min_value;
	int ver_in = -1;
	for (int i = 0; i < ver_num; i++) {
		if (solu[i] == 0 && a[i] > max_contribution) {
			max_contribution = a[i]; ver_in = i;
		}
	}
	solu[ver_in] = 1;//update
	int ver_out = vertex[k_num - 1];
	int x = address[ver_out];
	vertex[x] = ver_in; vertex[address[ver_in]] = ver_out;
	address[ver_out] = address[ver_in];
	address[ver_in] = x;
	update_para(ver_num + 1, ver_in);
	hashvalue = hashvalue + rand_vec[ver_in];
}

void K_Breakout_Local_Search(int time_limit) {
	lc = 0;
	wc = 20.0;
	L = L_MIN;
	prev_visit = 0;
	prev_decsent = 0;
	maxinc = 0;
	double now_time = (double)clock();
	clock_t run_time;
	run_time = (now_time - starting_time) / CLOCKS_PER_SEC;
	double ini_obj;
	double test_obj1 = 0, test_obj2 = 0;
	while (run_time < time_limit) {
		ini_K_Local_Search();
		double diff = (double)(best_obj - obj) / (double)best_obj;
		sumdiff += diff;
		countdiff++;
		if ((diff <= 0.5 * sumdiff / (double)countdiff) || diff < 0) {
			do {
				ini_obj = obj;
				ini_K_twoswap();//**************************************************************************************************************
				if (fabs(obj - ini_obj) <= 1e-6) { break; }
				ini_obj = obj;
				ini_K_Local_Search();
				if (fabs(obj - ini_obj) <= 1e-6) { break; }
				if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
			} while (obj > ini_obj);
		}
		if (obj > best_obj) {
			finishing_time = (double)clock();
			Save_bestsolu();
		}
		test_obj1 = obj;
		std::uniform_real_distribution<> dis(0.0, 1.0);
		double prob = dis(gen);
		if (prob < 0.5) {
			K_Delete();
			ini_K_Local_Search();
			K_Add();
		}
		else {
			K_Add();
			ini_K_Local_Search();
			K_Delete();
		}
		ini_K_Local_Search();
		diff = (double)(best_obj - obj) / (double)best_obj;
		sumdiff += diff;
		countdiff++;
		if ((diff <= 0.5 * sumdiff / (double)countdiff) || diff < 0) {
			do {
				ini_obj = obj;
				ini_K_twoswap();//**************************************************************************************************************
				if (fabs(obj - ini_obj) <= 1e-6) { break; }
				ini_obj = obj;
				ini_K_Local_Search();
				if (fabs(obj - ini_obj) <= 1e-6) { break; }
				if (((double)clock() - starting_time) / CLOCKS_PER_SEC > time_limit) { break; }
			} while (obj > ini_obj);
		}
		if (obj > best_obj) {
			finishing_time = (double)clock();
			Save_bestsolu();
		}
		test_obj2 = obj;
		prev_lo = obj;
		perturb_num++;
		DetermineJumpMagnitude();
		DeterminePerturbationType();
		now_time = (double)clock();
		run_time = (now_time - starting_time) / CLOCKS_PER_SEC;
	}
}

/************************************************* K CHANGE END ************************************************/
int main(int argc, char** argv) {
	int seed;
	string file1 = argv[1];
	time_limit = atof(argv[2]);
	sscanf(argv[3], "%d", &seed);
	gen.seed(seed * 10000);
	string instance_name = "Instance_HSPandMDP/" + file1;
	string resultfile = "RLBLS/";
	//paras:
	Pertype_num = 4; St2_Num = 4; St1_Num = 5;
	sigma = 0.5, gamma1 = 0.85, alpha = 0.2;
	st1_frame1 = 60, st1_frame2 = 90, st1_frame3 = 100;
	st2_frame1 = 10, st2_frame2 = 45, st2_frame3 = 80;
	InitPolicyStats();
	reward1 = 10.0, reward2 = 1.5, reward_para = 0.5;

	time_limit = 200;
	if (!file1.empty() && file1[0] == 'T') {
		instance_type = 1; cout << "hsp" << endl; ReadingHSPFile(instance_name);
		resultfile += "resultH/";
	}
	else {
		instance_type = 2; cout << "mdp" << endl; ReadingMDPFile(instance_name);
		resultfile += "resultM/";
	}
	wmin = max_value;
	for (int i = 0; i < ver_num - 1; i++) {
		for (int j = i + 1; j < ver_num; j++) {
			if (weight[i][j] < wmin) {
				wmin = weight[i][j];
			}
		}
	}
	resultfile += file1;
	ofstream outsol(resultfile, ios::app);
	if (!outsol) {
		cout << "can't open result file" << endl;
	}
	BuildArray();
	Set_Array_Memorybased();
	Set_Array_LS();
	L_MIN = ceil(0.01 * k_num);
	L_MAX = k_num;
	Iter = 0;
	L_step1 = 1; L_step2 = 1;
	SetVector();
	starting_time = (double)clock();
	Initialize();
	K_Breakout_Local_Search(time_limit);

	cout.setf(ios::fixed);
	outsol << std::fixed;
	if (instance_type == 1) {
		cout << "obj = " << std::setprecision(6) << best_obj;
		outsol << "obj = " << std::setprecision(6) << best_obj;
	}
	else {
		if (ver_num == 2500) {
			cout << "obj = " << std::setprecision(6) << 2.0 * best_obj;// << endl;
			outsol << "obj = " << std::setprecision(6) << 2.0 * best_obj;// << endl;
		}
		else {
			cout << "obj = " << std::setprecision(6) << best_obj;// << endl;
			outsol << "obj = " << std::setprecision(6) << best_obj;// << endl;
		}
	}
	cout << ", time = " << std::setprecision(2) << (finishing_time - starting_time) / CLOCKS_PER_SEC << endl;
	outsol << ", time = " << std::setprecision(2) << (finishing_time - starting_time) / CLOCKS_PER_SEC << endl;

}
