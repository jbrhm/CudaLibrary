#pragma once
#include <numeric>
#include <unordered_map>
#include <string>
#include <utility>
#include <chrono>
#include <vector>
#include <stdexcept>

// Performance Measurement utility
class LoopProfiler {
private:
    class TimingInfo {
    public:
        enum STATE {
            FINISHED = 0,
            STARTED = 1
        };

        STATE state;

        std::vector<double> times;

        std::chrono::time_point<std::chrono::system_clock> startTime;

        TimingInfo() : state{STATE::FINISHED} {}

        inline void start(){
            if(state == STATE::STARTED) throw std::runtime_error("Loop already started!");

            state = STATE::STARTED;

            startTime = std::chrono::system_clock::now();
        }

        inline void finish(){
            if(state == STATE::FINISHED) throw std::runtime_error("Loop already finished!");

            state = STATE::FINISHED;

            std::chrono::duration<double> elapsedSeconds = std::chrono::system_clock::now() - startTime;
            times.push_back(elapsedSeconds.count());
        }

        inline double avg() {
            double sum = 0;
            for(double d : times){
                sum += d;
            }
            return sum;
        }
    };

    std::unordered_map<std::string, TimingInfo> executionTimes;

public:
    LoopProfiler() : executionTimes{}{}

    inline void start(std::string const& name){
        executionTimes[name].start();
    }

    inline void finish(std::string const& name){
        executionTimes[name].finish();
    }

    inline double avg(std::string const& name){
        return executionTimes[name].avg();
    }
};