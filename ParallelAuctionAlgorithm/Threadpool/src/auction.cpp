#include <iostream>
#include <vector>
#include <limits>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include<string>
#include <cstring>
#include<fstream>
#include<ostream>
#include <cassert>
#include <stack>
#include <thread>
#include <mutex>
#include <list>

#include "include/thread-pool.hpp"

using namespace std;

#define inf numeric_limits<int>::max()
#define NUM_THREADS 4

vector<int> test_cases = {2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000};

/* Init variables */
vector<vector<int>> cost; //cost matrix
int n;// number of jobs and workers
vector<int> assignment(n, -1); //vector that safes the task that resource i is assigned to
vector<int> belong(n, -1); // vector that safes the resource that task j is assigned to
vector<double> prices(n, 1); // prices of tasks
double epsilon = 1.0;
stack<int> unassigned;
mutex mt_;
ThreadPool thpool_(NUM_THREADS);

void auctionRound_parallel(int id);

int auction_parallel()
{
	// Epsilon-scaling
	while(epsilon > 1.0/n)
	{	
		// reset arrays before new auction round
		for (int i=0; i<n; i++){
			assignment[i] = -1;
			belong[i] = -1;
		}
		// clear stack storing unassigned ressources before new auction round
		while (!unassigned.empty()){
			unassigned.pop();
		}
		// fill stack with unassigned ressources before new auction round
		for (size_t i = 0; i < assignment.size(); i++){
			if (assignment[i] == -1){
				unassigned.push(i);
			}
		}
		// auction round
		for (int i=0; i < NUM_THREADS; i++){
			thpool_.schedule([i]{
				auctionRound_parallel(i);
			}, i);
		}
		for (int i=0; i < NUM_THREADS; i++){
			thpool_.wait(i);
		}
		// scale epsilon after auction round
		epsilon = epsilon * .25;
	}

	// calculate total cost
	int res = 0;
	for (size_t i=0; i<assignment.size(); i++){
		res += cost[i][assignment[i]];
	}

	return res*(-1);
}

void auctionRound_parallel(int id)
{
	/* Compute the bids of each unassigned resource and assign them */
	while (true)
	{
		mt_.lock();
		// Check if unassigned resources are left
        if (unassigned.empty()) {
            mt_.unlock();
			break;
        }
		// Get unassigned resources from the top of the stack
        int bidder = unassigned.top();
        unassigned.pop();
        mt_.unlock();
		
		/* 
			Find a best task having maximum value and the corresponding value and find the second best value (offered by objects)
		*/

		double best_payoff = -inf;
		double sec_payoff = -inf;
		int best_task = -1;
		double price_task = -1;
			
		// Calculate best task for the unassigned resource
		for (int j = 0; j < n; j++)
		{
			double curVal = cost[bidder][j] - prices[j];
			if (curVal > best_payoff)
			{
				sec_payoff = best_payoff;
				best_payoff = curVal;
				best_task = j;
				price_task = prices[j];
			}
			else if (curVal > sec_payoff)
			{
				sec_payoff = curVal;
			}
		}

		/* Compute the bid */
		double amount = price_task + best_payoff - sec_payoff + epsilon;
		
		mt_.lock();
		// Assign resource to best task
		if (amount > prices[best_task]){
			int previous_owner = belong[best_task];
			if (previous_owner != -1) {
				// if someone already owns the item, kick him/her out
				assignment[previous_owner] = -1;
				unassigned.push(previous_owner);
			}
			/* Assign object j to id and update the price vector */
			assignment[bidder] = best_task;
			belong[best_task] = bidder;
			prices[best_task] = amount;
		}
		else {
			unassigned.push(bidder);
		}
		mt_.unlock();
	}
}

void create_matrix(vector<vector<int>>& mat, int size){
	srand (time(NULL)); 
	
	mat.resize(size);
	for(int i=0;i<size;++i){
        mat[i].resize(size);
        for(int j=0;j<size;++j){
            mat[i][j] = (rand() % size + 1) * (-1);
        }
    }
}

void reset(){
	assignment.resize(n, -1);
	belong.resize(n, -1);
	prices.resize(n, 1.0);
	epsilon = 1.0;
	
	for (int i=0; i<n; i++){
		assignment[i] = -1;
		belong[i] = -1;
		prices[i] = 1.0;
	}
}

double median(vector<double> v, int n)
{
    // Sort the vector
    sort(v.begin(), v.end());

    // Check if the number of elements is odd
    if (n % 2 != 0)
        return (double)v[n / 2];

    // If the number of elements is even, return the average
    // of the two middle elements
    return (double)(v[(n - 1) / 2] + v[n / 2]) / 2.0;
}

int main(int argc, char*argv[])
{
    if (argc != 3) {
        std::cerr << "Arguments must be presented as follows." << std::endl;
        std::cerr << "./name_test num_test_runs name_output_file" << std::endl;
        exit(1);
    }
	
	string output_path = "./data/";
	output_path.append(argv[2]);
    string filename_all_runs = output_path + ".txt";	
    string filename_mean_runs = output_path + "_mean.txt";
 		
	int num_test_runs = atoi(argv[1]);
	
	ofstream out_file_all;
    out_file_all.open (filename_all_runs);
	out_file_all<<"Auction Parallel ThreadPool ("<<NUM_THREADS<<" Threads): \n";
	out_file_all<<"Size\t\t\t\tCost\t\t\t\tTotal Time in s\t\t\tTotal CPU time in ms\n";
	
	ofstream out_file_mean;
    out_file_mean.open (filename_mean_runs);
	out_file_mean<<"Auction Parallel ThreadPool ("<<NUM_THREADS<<" Threads, "<<num_test_runs<<" runs): \n";
	out_file_mean<<"Size\t\t\t\tMean Time in s\t\t\tMean CPU time in ms\t\t\tMedian Time in s\t\t\tMedian CPU Time in ms\n";
	
	for (int& t : test_cases){ // n is the number of jobs and workers
		double sum_timing = 0;
		double sum_time = 0;

		vector<double> times(test_cases.size());
        vector<double> timings(test_cases.size());
		
		for (int r=0; r < num_test_runs; r++){
			n = t;
			cout<<"Test Case:\t"<<n<<endl;
			/* init cost matrix for test case */
			create_matrix(cost, n);
		
			reset();

			int res; // result of algorithm
			clock_t start, end; // start and end time of algorithm

			/* Begin Time */
			auto t1 = std::chrono::high_resolution_clock::now();
			start = clock();

			res=auction_parallel();  
			end = clock();
	
			/* End Time */
			auto t2 = std::chrono::high_resolution_clock::now();
			double timing = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
			double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
			
			times.push_back(time);
			timings.push_back(timing);
				
			sum_timing += timing;
			sum_time += time;

			out_file_all<<n<<"\t\t\t\t"<<res<<"\t\t\t\t"<<timing / 1000.0<<"\t\t\t\t"<<time<<"\n";
		}
		
		double mean_timing = sum_timing / num_test_runs;
		double mean_time = sum_time / num_test_runs;
		
		double median_timing = median(timings, timings.size());
		double median_time = median(times, times.size());
		
		out_file_mean<<n<<"\t\t\t\t"<<mean_timing / 1000.0<<"\t\t\t\t"<<mean_time<<"\t\t\t\t"<<median_timing / 1000.0<<"\t\t\t\t"<<median_time<<"\n";
	}
	out_file_all.close();
	out_file_mean.close();
}
