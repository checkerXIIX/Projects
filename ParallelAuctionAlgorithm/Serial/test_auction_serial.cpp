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
#include <list>

using namespace std;

vector<int> test_cases = {2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000};

#define inf numeric_limits<int>::max()


void auctionRound(vector<vector<int>> &cost, int n, vector<int>& assignment, vector<int>& belong, vector<double>& prices, double epsilon);

int auction(vector<vector<int>> &cost, int n)
{
	vector<int> assignment(n, -1); //vector that safes the task that resource i is assigned to
	vector<int> belong(n, -1); // vector that safes the resource that task j is assigned to
	vector<double> prices(n, 1); // prices of tasks
	double epsilon = 1.0;

	// Epsilon-scaling
	while(epsilon > 1.0/n)
	{
		// reset arrays before new auction round
		for (int i=0; i<n; i++){
			assignment[i] = -1;
			belong[i] = -1;
		}
		// auction round until all bidders are assigned
		while (find(assignment.begin(), assignment.end(), -1) != assignment.end())
		{
			auctionRound(cost, n, assignment, belong, prices, epsilon);

		}
		// scale epsilon after auction round
		epsilon = epsilon * .25;
	}
	
	// calculate total cost
	int res = 0;
	for (int i=0; i<assignment.size(); i++){
		res += cost[i][assignment[i]];
	}

	return res*(-1);
}

void auctionRound(vector<vector<int>> &cost, int n, vector<int>& assignment, vector<int>& belong, vector<double>& prices, double epsilon)
{
	struct bid{      
		int bidder;		 // Id of bidder
		int task;      	 // Id of task that bidder bidded for
		double amount;   // Amount of bid of bidder
	};      
	
	vector<bid> bidders; // vector to safe the bids of each unassigned resource

	/* Compute the bids of each unassigned resource */
	for (int i = 0; i < assignment.size(); i++)
	{
		if (assignment[i] == -1)
		{
			// Calculate the best and second-best value of each task to this bidder
			double best_payoff = -inf;
			double sec_payoff = -inf;
			int best_task;
			double price_task;
			for (int j = 0; j < n; j++)
			{
				double curVal = cost[i][j] - prices[j];
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

			/* Creating new bid */
			bid bid_i;
			bid_i.bidder = i;
			bid_i.amount = price_task + best_payoff - sec_payoff + epsilon;
			bid_i.task = best_task;

			/* Stores the bidding info for future use */
			bidders.push_back(bid_i);
		}
	}

	/* 
		Assignment Phase of the Auction Iteration:
		Each task that has received a bid, increases its price to the highest bid received.
	*/
	for (int j = 0; j < n; j++)
	{
		vector<bid> bids_task;
		for (int i=0; i < bidders.size(); i++){
			if (bidders[i].task == j){
				bids_task.push_back(bidders[i]);
			}
		}
		
		if (bids_task.size() != 0)
		{	
			// Search for the highest bid received
			double highest_bid = -inf; // amount of highest bid received by the task
			int id; //id of new bidder assigned to task j
			for (int i = 0; i < bids_task.size(); i++)
			{
				bid bid_i = bids_task[i];
				double curVal = bid_i.amount;
				if (curVal > highest_bid)
				{
					highest_bid = curVal;
					id = bid_i.bidder;
				}
			}

			/* Remove assignment (if j was assigned to another i before) */
			int previous_owner = belong[j];
			if (previous_owner != -1) {
				// if someone already owns the item, kick him/her out
				assignment[previous_owner] = -1;
			}
			/* Assign task j to id and update the price vector */
			assignment[id] = j;
			belong[j] = id;
			prices[j] = highest_bid;
		}
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
	out_file_all<<"Auction Serial: \n";
	out_file_all<<"Size\t\t\t\tCost\t\t\t\tTotal Time in s\t\t\tTotal CPU time in ms\n";
	
	ofstream out_file_mean;
    out_file_mean.open (filename_mean_runs);
	out_file_mean<<"Auction Serial ("<<num_test_runs<<" runs): \n";
	out_file_mean<<"Size\t\t\t\tMean Time in s\t\t\tMean CPU time in ms\t\t\tMedian Time in s\t\t\tMedian CPU Time in ms\n";
	
	for (int& n : test_cases){ // n is the number of jobs and workers
		double sum_timing = 0;
		double sum_time = 0;
		vector<double> times(test_cases.size());
        vector<double> timings(test_cases.size());

		for (int r=0; r < num_test_runs; r++){
			cout<<"Test Case:\t"<<n<<endl;
			/* init cost matrix for test case */
			vector<vector<int>> cost; //cost matrix
			create_matrix(cost, n);

			int res; // result of algorithm
			clock_t start, end; // start and end time of algorithm

			/* Begin Time */
			auto t1 = std::chrono::high_resolution_clock::now();
			start = clock();

			res=auction(cost, n); 
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
