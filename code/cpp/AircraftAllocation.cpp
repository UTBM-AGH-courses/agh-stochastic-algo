#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <string>
#include <thread>

class Data
{
public:
    Data() {}
    ~Data() {}

    std::vector<int> m_vAircraftAllocation;

    std::vector<float> m_vCosts;
    std::vector<float> m_vRevenueLost;
    std::vector<float> m_vPassengerDemand;

    std::vector<int> m_vAircraftAvailable;
    std::vector<int> m_vAircraftCapacity;

    std::vector<std::vector<int>> m_bAircraftRouteAllocation; // [routeNumber][aircraftType]

    std::vector<int> m_vPassengerTurnedAway;

    bool m_bVerbose = false;

    void LoadDataSet_1()
    {
        m_vCosts = {18, 21, 18, 16, 10, 15, 16, 14, 9, 10, 9, 6, 17, 16, 17, 15, 10, 18, 16, 12};
        m_vRevenueLost = {13, 13, 7, 7, 1};
        m_vAircraftCapacity = {16, 15, 28, 23, 81, 10, 14, 15, 57, 5, 7, 29, 9, 11, 22, 17, 55, 16, 20, 18};
        m_vAircraftAvailable = {10, 19, 25, 15};

        m_bAircraftRouteAllocation = {{0, 5, 10, 15}, {1, 6, 11, 16}, {2, 7, 12, 17}, {3, 8, 13, 18}, {4, 9, 14, 19}};

        m_vPassengerDemand = {250, 100, 180, 100, 600};
    }

    void LoadDataSet_2()
    {
        m_vCosts = {12, 2, 43, 32, 20, 20, 34, 63, 10, 30, 30, 10, 40, 6, 10, 19, 20, 12, 34, 87};
        m_vRevenueLost = {13, 20, 7, 7, 15};
        m_vAircraftCapacity = {16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 23, 23, 23, 23, 23};
        m_vAircraftAvailable = {10, 19, 25, 16};

        m_bAircraftRouteAllocation = {{0, 5, 10, 15}, {1, 6, 11, 16}, {2, 7, 12, 17}, {3, 8, 13, 18}, {4, 9, 14, 19}};

        m_vPassengerDemand = {800, 900, 700, 650, 380};
    }

    float CostEvaluation(const std::vector<int> &p_vAircraftAllocation)
    {
        if (!CheckAircraftAllocationIntegrity(p_vAircraftAllocation))
        {
            std::cout << " Wrong Size Data ";
        }

        float l_fResult = 0;

        // Calcul total operating cost
        for (int index = 0; index < p_vAircraftAllocation.size(); ++index)
        {
            l_fResult += m_vCosts[index] * p_vAircraftAllocation[index];
        }

        if (m_bVerbose)
            std::cout << " total route operating cost : " << l_fResult << std::endl;

        // calcul number of passenger turned away per route
        std::vector<float> l_vPassengersTurnedAway;

        // for each route
        for (int index = 0; index < m_vRevenueLost.size(); ++index)
        {
            float l_fPassengerTA = m_vPassengerDemand[index];

            // for each aircraft type in route
            for (auto idAircraftRoute : m_bAircraftRouteAllocation[index])
            {
                l_fPassengerTA -= m_vAircraftCapacity[idAircraftRoute] * p_vAircraftAllocation[idAircraftRoute];
            }

            if (l_fPassengerTA < 0) // more place than expected passengers
                l_vPassengersTurnedAway.push_back(0);
            else // less place
                l_vPassengersTurnedAway.push_back(l_fPassengerTA);

            if (m_bVerbose)
                std::cout << "passengers TA on route " << index + 1 << " : " << (l_fPassengerTA < 0 ? 0 : l_fPassengerTA) << std::endl;
        }

        // calcul revenue lost
        for (int route = 0; route < 5; ++route)
        {
            l_fResult += m_vRevenueLost[route] * l_vPassengersTurnedAway[route];

            if (m_bVerbose)
                std::cout << "revenue lost on route " << route + 1 << " : " << m_vRevenueLost[route] << "*" << l_vPassengersTurnedAway[route] << " = " << m_vRevenueLost[route] * l_vPassengersTurnedAway[route] << std::endl;
        }

        if (m_bVerbose)
            std::cout << "result : " << l_fResult << std::endl;

        return l_fResult;
    }

    std::vector<int> GenerateRandomAllocation()
    {
        std::vector<int> l_vRandomAllocation(20, 0);

        // For each aircraft type
        for (int aircraftType = 0; aircraftType < m_vAircraftAvailable.size(); ++aircraftType)
        {
            int l_iAircraftLeft = m_vAircraftAvailable[aircraftType];

            int l_iImproveRandom = 2;

            // For each route
            for (auto routeAirplane : m_bAircraftRouteAllocation)
            {
                if (l_iAircraftLeft != 0)
                {
                    int l_iAircraftAttribution = GenerateRandomNumber(0, l_iAircraftLeft / l_iImproveRandom);
                    l_iAircraftLeft -= l_iAircraftAttribution;
                    l_vRandomAllocation[routeAirplane[aircraftType]] = l_iAircraftAttribution;
                }
            }

            // if some airplane is still available, put it randomly in one route
            if (l_iAircraftLeft != 0)
                l_vRandomAllocation[m_bAircraftRouteAllocation[GenerateRandomNumber(0, 3)][aircraftType]] += l_iAircraftLeft;
        }

        //printAllocation(l_vRandomAllocation);

        return l_vRandomAllocation;
    }

    std::vector<std::vector<int>> GenerateNeighbors(const std::vector<int> &p_vInitAllocation, const int &l_iAircraftType, const int &p_iNeighborsNumber = 4)
    {
        std::vector<std::vector<int>> l_vNeighbors;

        int rd;

        for (int count = 0; count < p_iNeighborsNumber; ++count)
        {
            rd = GenerateRandomNumber(0, 2);

            if (rd == 0)
            {
                l_vNeighbors.push_back(SingleRandomPermutation(p_vInitAllocation, l_iAircraftType));
            }
            else if (rd == 1)
            {
                l_vNeighbors.push_back(RandomPlusOneMinusOne(p_vInitAllocation, l_iAircraftType));
            }
            else if (rd == 2)
            {
                l_vNeighbors.push_back(WholeNewColumn(p_vInitAllocation, l_iAircraftType));
            }
        }

        return l_vNeighbors;
    }

    void printAllocation(const std::vector<int> &p_vAllocation)
    {
        int route = 1;

        // For each aircraft type
        for (int aircraftType = 0; aircraftType < m_vAircraftAvailable.size(); ++aircraftType)
        {
            std::cout << " Aircraft type : " << aircraftType + 1 << std::endl;

            for (auto routeAirplane : m_bAircraftRouteAllocation)
            {
                std::cout << "x" << routeAirplane[aircraftType] + 1 << " : " << p_vAllocation[routeAirplane[aircraftType]] << std::endl;
            }
        }
    }

private:
    bool CheckAircraftAllocationIntegrity(const std::vector<int> &p_vAircraftAllocation)
    {
        if (p_vAircraftAllocation.size() != m_vAircraftCapacity.size())
            return false;

        return true;
    }

    // Generate random number between min and max
    float GenerateRandomNumber(int min, int max)
    {
        // https://en.cppreference.com/w/cpp/numeric/random
        // Seed with a real random value, if available
        std::random_device r;

        // Choose a random mean between min and max
        std::default_random_engine e1(r());
        std::uniform_int_distribution<int> uniform_dist(min, max);
        float number = ((float)uniform_dist(e1));
        return number;
    }

    std::vector<int> SingleRandomPermutation(const std::vector<int> &p_vAllocation, const int aircraftType)
    {
        std::vector<int> l_vOutput = p_vAllocation;

        // Compute index
        int l_iIndexToPermuteFirst = GenerateRandomNumber(aircraftType * 5, (aircraftType * 5) + 4);
        int l_iIndexToPermuteSecond = l_iIndexToPermuteFirst;

        while (l_iIndexToPermuteFirst == l_iIndexToPermuteSecond)
            l_iIndexToPermuteSecond = GenerateRandomNumber(aircraftType * 5, (aircraftType * 5) + 4);

        // Permute
        int temp = l_vOutput[l_iIndexToPermuteFirst];
        l_vOutput[l_iIndexToPermuteFirst] = l_vOutput[l_iIndexToPermuteSecond];
        l_vOutput[l_iIndexToPermuteSecond] = temp;

        return l_vOutput;
    }

    std::vector<int> WholeNewColumn(const std::vector<int> &p_vAllocation, const int aircraftType)
    {
        std::vector<int> l_vOutput = p_vAllocation;

        int l_iAircraftLeft = m_vAircraftAvailable[aircraftType];

        int l_iImproveRandom = 2;

        // For each airplane type in route
        for (auto routeAirplane : m_bAircraftRouteAllocation)
        {
            if (l_iAircraftLeft != 0)
            {
                int l_iAircraftAttribution = GenerateRandomNumber(0, l_iAircraftLeft / l_iImproveRandom);
                l_iAircraftLeft -= l_iAircraftAttribution;
                l_vOutput[routeAirplane[aircraftType]] = l_iAircraftAttribution;
            }
        }

        // if some airplane is still available, put it randomly in one route
        if (l_iAircraftLeft != 0)
            l_vOutput[m_bAircraftRouteAllocation[GenerateRandomNumber(0, 3)][aircraftType]] += l_iAircraftLeft;

        return l_vOutput;
    }

    std::vector<int> RandomPlusOneMinusOne(const std::vector<int> &p_vAllocation, const int aircraftType)
    {
        std::vector<int> l_vOutput = p_vAllocation;

        // Compute index
        int l_iIndexToPermuteFirst = GenerateRandomNumber(aircraftType * 5, (aircraftType * 5) + 4);

        bool isValid = false;
        // Error check
        for (int index = aircraftType * 5; index <= (aircraftType * 5) + 4; ++index)
        {
            if (index != l_iIndexToPermuteFirst)
                if (p_vAllocation[index] > 0)
                    isValid = true;
        }

        int l_iIndexToPermuteSecond = l_iIndexToPermuteFirst;

        if (isValid)
        {
            while (l_iIndexToPermuteFirst == l_iIndexToPermuteSecond || l_vOutput[l_iIndexToPermuteSecond] == 0)
                l_iIndexToPermuteSecond = GenerateRandomNumber(aircraftType * 5, (aircraftType * 5) + 4);

            // apply
            l_vOutput[l_iIndexToPermuteFirst] += 1;
            l_vOutput[l_iIndexToPermuteSecond] -= 1;
        }
        else // only one positive value
        {
            while (l_iIndexToPermuteFirst == l_iIndexToPermuteSecond)
                l_iIndexToPermuteSecond = GenerateRandomNumber(aircraftType * 5, (aircraftType * 5) + 4);

            // apply
            l_vOutput[l_iIndexToPermuteFirst] -= 1;
            l_vOutput[l_iIndexToPermuteSecond] += 1;
        }

        return l_vOutput;
    }
};

class SimulatedAnnealing
{

public:
    SimulatedAnnealing() {}
    SimulatedAnnealing(float p_fTemperature, float p_fDecreaseFactor) : m_fTemperature(p_fTemperature), m_fDecreaseFactor(p_fDecreaseFactor), m_iCountStatic(0), m_iTryNumberCount(0) {}

    ~SimulatedAnnealing() {}

    float m_fTemperature;

    float m_fDecreaseFactor;

    int m_iCountStatic;

    const int m_iTryNumber = 3;
    int m_iTryNumberCount;

    int m_iStepNumber = 4; // equals to the number of aircraft type

    float ApplyAlgorithm(Data &p_Data)
    {
        float l_fTemperature = m_fTemperature; // Set the temperature value

        // initiate first candidate and its cost
        std::vector<int> l_vCurrentCandidate = p_Data.GenerateRandomAllocation();
        float l_fCurrentCost = p_Data.CostEvaluation(l_vCurrentCandidate);

        std::vector<float> l_vCostHistory{l_fCurrentCost}; // to store the cost history

        std::vector<std::vector<int>> l_vNeighbors; // variable declaration

        int l_iLoopCount = 0;
        int l_iCostLimit = 18700;

        // Loop on temperature condition
        while (l_fTemperature > 0.1)
        {
            // Each Step before decreasing the temperature
            for (int step = 0; step < m_iStepNumber; ++step)
            {
                // Step 1 : Generate Neighbors
                l_vNeighbors = p_Data.GenerateNeighbors(l_vCurrentCandidate, step);

                // Step 2 : Find Minimum cost neighbor
                std::vector<int> l_vMinimunCostNeighbor;
                float l_fCost = FLT_MAX;

                for (auto NeighBorsCandidate : l_vNeighbors)
                {
                    float l_fNewCost = p_Data.CostEvaluation(NeighBorsCandidate);
                    // Loop Count
                    l_iLoopCount++;

                    if (l_fNewCost < l_fCost)
                    {
                        l_vMinimunCostNeighbor = NeighBorsCandidate;
                        l_fCost = l_fNewCost;
                    }
                }

                // Step 3 : Apply SA condition

                float l_fNeighborsCandidateCost = p_Data.CostEvaluation(l_vMinimunCostNeighbor);
                float l_fCostDifference = l_fNeighborsCandidateCost - l_fCurrentCost;

                // If it's a better value choose it
                if (l_fCostDifference < 0)
                {
                    l_vCurrentCandidate = l_vMinimunCostNeighbor;
                    l_fCurrentCost = l_fNeighborsCandidateCost;
                }
                // Condition with the temperature value
                else if (GenerateRandomNumber(0, 100) < exp((-l_fCostDifference) / l_fTemperature))
                {
                    l_vCurrentCandidate = l_vMinimunCostNeighbor;
                    l_fCurrentCost = l_fNeighborsCandidateCost;
                }
            }

            // Cost History
            l_vCostHistory.push_back(l_fCurrentCost);

            // Decrease the temperature
            l_fTemperature *= (1 - m_fDecreaseFactor);
        }

        // Generate csv for cost history
        std::ofstream myfile;
        myfile.open("result_.csv");
        myfile << "Cost History.\n";
        myfile << "T = " << m_fTemperature << ",\n";
        myfile << "DFactor = " << m_fDecreaseFactor << ",\n";
        myfile << "FinalCost = " << l_vCostHistory.back() << ",\n";
        for (auto cost : l_vCostHistory)
            myfile << cost << ",\n";
        myfile.close();

        // Print Solution
        p_Data.printAllocation(l_vCurrentCandidate);

        return l_vCostHistory.back();
    }

private:
    // Generate random number between min and max and divided it by 100
    float GenerateRandomNumber(int min, int max)
    {
        // https://en.cppreference.com/w/cpp/numeric/random
        // Seed with a real random value, if available
        std::random_device r;

        // Choose a random mean between 0 and 100
        std::default_random_engine e1(r());
        std::uniform_int_distribution<int> uniform_dist(min, max);
        float number = ((float)uniform_dist(e1)) / 100;

        return number;
    }
};

int main()
{
    std::cout << "########## Aircraft Allocation Problem ##########\n";

    Data l_Data;
    SimulatedAnnealing l_SA(999, 0.01);

    l_Data.LoadDataSet_2();

    l_SA.ApplyAlgorithm(l_Data);

    std::cout << "the end";
}
