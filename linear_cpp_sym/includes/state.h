//
// Created by fenix on 5/14/20.
//

#ifndef LINEAR_CPP_SYM_STATE_H
#define LINEAR_CPP_SYM_STATE_H


struct States{
    int normal, infected, patient, dead;
    double get_normal, get_infected, get_patient, get_dead;
};


class State {
private:
    int id, next_state_id;
    std::string repr;
    double next_prob;

public:
    State(int new_id, std::string new_repr, int next_state, float new_prob) {
        id = new_id;
        repr = new_repr;
        next_prob = new_prob;
        next_state_id = next_state;
    }

    bool if_equal(State other){
        return id == other.id;
    }

    int get_id(){
        return id;
    }

    std::string get_repr(){
        return repr;
    }

    double get_prob(){
        return next_prob;
    }

    int get_next_state_id(){
        return next_state_id;
    }
};


#endif //LINEAR_CPP_SYM_STATE_H
