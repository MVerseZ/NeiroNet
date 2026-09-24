#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ===== Identifiers ===== */
typedef size_t NeuronId;
typedef size_t SynapseId;

/* ============================================================
 *                    TUNING CONSTANTS
 * ============================================================ */

/* --- Simulation --- */
#define SIM_DURATION          200.0   /* total simulated time (ms) */
#define SIM_DT                1.0     /* time step (ms) */
#define SIM_STEPS             200     /* number of steps */
#define RASTER_WIDTH          100     /* ASCII raster width (chars) */

/* --- Network composition --- */
#define NET_N_EXCITATORY      20      /* number of excitatory neurons */
#define NET_N_INHIBITORY      8       /* number of inhibitory neurons */
#define NET_N_MODULATORY      4       /* number of modulatory neurons */
#define NET_SYNAPSES_PER_NODE 3       /* average outgoing synapses per neuron */

/* --- Synaptic transmission --- */
#define MAX_DELAY_STEPS       64      /* max synaptic delay in steps */
#define SYNAPTIC_GAIN         30.0    /* synaptic input multiplier */
#define SYNAPTIC_DELAY_MIN    1.0     /* min synaptic delay (ms) */
#define SYNAPTIC_DELAY_MAX    5.0     /* max synaptic delay (ms) */

/* --- Noise --- */
#define NOISE_AMPLITUDE       2.0     /* +/- mV per step */

/* --- Neuron defaults --- */
#define V_RESTING             (-70.0) /* resting potential (mV) */
#define V_THRESHOLD_EXC       (-55.0) /* threshold: excitatory */
#define V_THRESHOLD_INH       (-60.0) /* threshold: inhibitory */
#define V_THRESHOLD_MOD       (-50.0) /* threshold: modulatory */
#define V_RESET_EXC           (-70.0) /* reset: excitatory */
#define V_RESET_INH           (-75.0) /* reset: inhibitory */
#define V_RESET_MOD           (-65.0) /* reset: modulatory */
#define REFRACTORY_EXC        2.0     /* refractory: excitatory (ms) */
#define REFRACTORY_INH        1.0     /* refractory: inhibitory (ms) */
#define REFRACTORY_MOD        5.0     /* refractory: modulatory (ms) */

/* --- External currents (per neuron type) --- */
#define I_EXT_EXC_BASE        2.5     /* excitatory: base external current */
#define I_EXT_EXC_RANGE       2.0     /* excitatory: random range */
#define I_EXT_INH_BASE        1.0     /* inhibitory: base external current */
#define I_EXT_INH_RANGE       1.0     /* inhibitory: random range */
#define I_EXT_MOD_BASE        1.5     /* modulatory: base external current */
#define I_EXT_MOD_RANGE       1.0     /* modulatory: random range */

/* --- Plasticity (STDP) --- */
#define STDP_ENABLED          true
#define STDP_LEARNING_RATE    0.01    /* learning rate */
#define STDP_TAU              20.0    /* time constant (ms) */
#define STDP_MAX_WEIGHT       1.0     /* upper clamp */
#define STDP_MIN_WEIGHT      (-1.0)   /* lower clamp */

/* --- Synapse weight ranges (random init) --- */
#define W_EXC_MIN             0.2     /* excitatory: min weight */
#define W_EXC_RANGE           0.5     /* excitatory: random range */
#define W_INH_MIN            (-0.2)   /* inhibitory: min weight (negative) */
#define W_INH_RANGE          (-0.5)   /* inhibitory: random range (negative) */
#define W_MOD_MIN             0.1     /* modulatory: min weight */
#define W_MOD_RANGE           0.2     /* modulatory: random range */

/* ===== Enums ===== */
typedef enum {
    SYNAPSE_EXCITATORY,
    SYNAPSE_INHIBITORY,
    SYNAPSE_MODULATORY
} SynapseType;

typedef enum {
    NEURON_EXCITATORY,
    NEURON_INHIBITORY,
    NEURON_MODULATORY
} NeuronType;

/* ===== Forward declarations ===== */
typedef struct Neuron Neuron;
typedef struct Synapse Synapse;
typedef struct Graph Graph;

/* ===== Axon ===== */
typedef struct Axon {
    double length;
    double diameter;
    double conduction_velocity;
    bool   is_myelinated;
    double myelin_thickness;
    size_t num_terminals;
    double resting_potential;
    double threshold;

    Synapse **out_synapses;
    size_t    out_count;
    size_t    out_capacity;
} Axon;

/* ===== Synapse ===== */
typedef struct Synapse {
    SynapseId id;

    NeuronId presynaptic_id;
    NeuronId postsynaptic_id;

    SynapseType type;
    double weight;
    double delay;
    double neurotransmitter_concentration;

    bool   is_plastic;
    double learning_rate;
    double last_activation;

    double pre_trace;
    double post_trace;

    double *delay_line;
    int     delay_line_size;

    Neuron *presynaptic;
    Neuron *postsynaptic;
} Synapse;

/* ===== Neuron ===== */
struct Neuron {
    NeuronId id;
    NeuronType type;
    double membrane_potential;
    double firing_rate;

    double threshold;
    double reset_potential;
    double refractory_period;
    double last_spike_time;
    bool   is_refractory;

    double external_current;

    double spike_trace;

    double *spike_times;
    size_t  spike_count;
    size_t  spike_capacity;

    Axon     *axon;
    Synapse **in_synapses;
    size_t    in_count;
    size_t    in_capacity;
};

/* ===== Graph ===== */
struct Graph {
    Neuron  **neurons;
    size_t    neuron_count;
    size_t    neuron_capacity;

    Synapse **synapses;
    size_t    synapse_count;
    size_t    synapse_capacity;

    NeuronId  next_neuron_id;
    SynapseId next_synapse_id;

    double current_time;
    double dt;
    long   step;
    double sim_duration;
};

/* ============================================================
 *                    GRAPH FUNCTIONS
 * ============================================================ */

Graph *graph_create(double sim_duration, double dt) {
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) return NULL;
    g->next_neuron_id  = 0;
    g->next_synapse_id = 0;
    g->current_time = 0.0;
    g->sim_duration = sim_duration;
    g->dt = dt;
    g->step = 0;
    return g;
}

void graph_free(Graph *g) {
    if (!g) return;
    for (size_t i = 0; i < g->synapse_count; i++) {
        free(g->synapses[i]->delay_line);
        free(g->synapses[i]);
    }
    free(g->synapses);
    for (size_t i = 0; i < g->neuron_count; i++) {
        Neuron *n = g->neurons[i];
        free(n->spike_times);
        if (n->axon) {
            free(n->axon->out_synapses);
            free(n->axon);
        }
        free(n->in_synapses);
        free(n);
    }
    free(g->neurons);
    free(g);
}

Neuron *graph_add_neuron(Graph *g, NeuronType type) {
    if (!g) return NULL;
    if (g->neuron_count == g->neuron_capacity) {
        size_t new_cap = g->neuron_capacity ? g->neuron_capacity * 2 : 16;
        Neuron **tmp = realloc(g->neurons, new_cap * sizeof(Neuron *));
        if (!tmp) return NULL;
        g->neurons = tmp;
        g->neuron_capacity = new_cap;
    }
    Neuron *n = calloc(1, sizeof(Neuron));
    if (!n) return NULL;
    n->id = g->next_neuron_id++;
    n->type = type;

    switch (type) {
        case NEURON_EXCITATORY:
            n->membrane_potential = V_RESTING;
            n->threshold = V_THRESHOLD_EXC;
            n->reset_potential = V_RESET_EXC;
            n->refractory_period = REFRACTORY_EXC;
            break;
        case NEURON_INHIBITORY:
            n->membrane_potential = V_RESTING;
            n->threshold = V_THRESHOLD_INH;
            n->reset_potential = V_RESET_INH;
            n->refractory_period = REFRACTORY_INH;
            break;
        case NEURON_MODULATORY:
            n->membrane_potential = V_RESET_MOD;
            n->threshold = V_THRESHOLD_MOD;
            n->reset_potential = V_RESET_MOD;
            n->refractory_period = REFRACTORY_MOD;
            break;
    }
    n->firing_rate = 0.0;
    n->last_spike_time = -1000.0;
    n->is_refractory = false;
    n->external_current = 0.0;
    n->spike_trace = 0.0;
    n->spike_times = NULL;
    n->spike_count = 0;
    n->spike_capacity = 0;
    n->axon = NULL;
    n->in_synapses = NULL;
    n->in_count = 0;
    n->in_capacity = 0;
    g->neurons[g->neuron_count++] = n;
    return n;
}

Axon *neuron_add_axon(Neuron *n) {
    if (!n) return NULL;
    Axon *a = calloc(1, sizeof(Axon));
    if (!a) return NULL;
    a->length = 0.0;
    a->diameter = 1.0;
    a->conduction_velocity = 1.0;
    a->is_myelinated = false;
    a->myelin_thickness = 0.0;
    a->num_terminals = 0;
    a->resting_potential = V_RESTING;
    a->threshold = V_THRESHOLD_EXC;
    a->out_synapses = NULL;
    a->out_count = 0;
    a->out_capacity = 0;
    n->axon = a;
    return a;
}

static void synapse_alloc_delay_line(Synapse *s, double dt) {
    int steps = (int)ceil(s->delay / dt) + 1;
    if (steps < 1) steps = 1;
    if (steps > MAX_DELAY_STEPS) steps = MAX_DELAY_STEPS;
    s->delay_line_size = steps;
    s->delay_line = calloc((size_t)steps, sizeof(double));
}

Synapse *graph_add_synapse(Graph *g, Neuron *pre, Neuron *post,
                           SynapseType type, double weight, double delay) {
    if (!g || !pre || !post) return NULL;
    if (g->synapse_count == g->synapse_capacity) {
        size_t new_cap = g->synapse_capacity ? g->synapse_capacity * 2 : 32;
        Synapse **tmp = realloc(g->synapses, new_cap * sizeof(Synapse *));
        if (!tmp) return NULL;
        g->synapses = tmp;
        g->synapse_capacity = new_cap;
    }
    Synapse *s = calloc(1, sizeof(Synapse));
    if (!s) return NULL;
    s->id = g->next_synapse_id++;
    s->presynaptic_id  = pre->id;
    s->postsynaptic_id = post->id;
    s->type = type;
    s->weight = weight;
    s->delay = delay;
    s->neurotransmitter_concentration = 0.0;
    s->is_plastic = STDP_ENABLED;
    s->learning_rate = STDP_LEARNING_RATE;
    s->last_activation = 0.0;
    s->pre_trace = 0.0;
    s->post_trace = 0.0;
    s->presynaptic  = pre;
    s->postsynaptic = post;
    s->delay_line = NULL;
    s->delay_line_size = 0;
    synapse_alloc_delay_line(s, g->dt);
    g->synapses[g->synapse_count++] = s;

    if (pre->axon) {
        Axon *a = pre->axon;
        if (a->out_count == a->out_capacity) {
            size_t new_cap = a->out_capacity ? a->out_capacity * 2 : 4;
            Synapse **tmp = realloc(a->out_synapses, new_cap * sizeof(Synapse *));
            if (!tmp) return s;
            a->out_synapses = tmp;
            a->out_capacity = new_cap;
        }
        a->out_synapses[a->out_count++] = s;
    }
    if (post->in_count == post->in_capacity) {
        size_t new_cap = post->in_capacity ? post->in_capacity * 2 : 4;
        Synapse **tmp = realloc(post->in_synapses, new_cap * sizeof(Synapse *));
        if (!tmp) return s;
        post->in_synapses = tmp;
        post->in_capacity = new_cap;
    }
    post->in_synapses[post->in_count++] = s;
    return s;
}

Neuron *graph_find_neuron(Graph *g, NeuronId id) {
    if (!g) return NULL;
    for (size_t i = 0; i < g->neuron_count; i++)
        if (g->neurons[i]->id == id) return g->neurons[i];
    return NULL;
}

/* ============================================================
 *                    SPIKE SIMULATION
 * ============================================================ */

void neuron_record_spike(Neuron *n, double t) {
    if (n->spike_count == n->spike_capacity) {
        size_t new_cap = n->spike_capacity ? n->spike_capacity * 2 : 32;
        double *tmp = realloc(n->spike_times, new_cap * sizeof(double));
        if (!tmp) return;
        n->spike_times = tmp;
        n->spike_capacity = new_cap;
    }
    n->spike_times[n->spike_count++] = t;
}

static void synapse_schedule_spike(Synapse *s, double amount, long current_step) {
    if (!s->delay_line || s->delay_line_size <= 0) return;
    long idx = (current_step + (long)round(s->delay)) % s->delay_line_size;
    s->delay_line[idx] += amount;
}

static double synapse_consume_delayed(Synapse *s, long current_step) {
    if (!s->delay_line || s->delay_line_size <= 0) return 0.0;
    long idx = current_step % s->delay_line_size;
    double v = s->delay_line[idx];
    s->delay_line[idx] = 0.0;
    return v;
}

void graph_step(Graph *g) {
    if (!g) return;

    /* 1. Deliver delayed signals */
    for (size_t i = 0; i < g->synapse_count; i++) {
        Synapse *s = g->synapses[i];
        double amount = synapse_consume_delayed(s, g->step);
        if (amount != 0.0) {
            Neuron *post = s->postsynaptic;
            if (s->type == SYNAPSE_MODULATORY)
                post->membrane_potential += amount * 0.5;
            else
                post->membrane_potential += amount;
        }
    }

    /* 2. Refractory update */
    for (size_t i = 0; i < g->neuron_count; i++) {
        Neuron *n = g->neurons[i];
        if (n->is_refractory &&
            (g->current_time - n->last_spike_time) >= n->refractory_period)
            n->is_refractory = false;
    }

    /* 3. Leak + external drive + noise */
    for (size_t i = 0; i < g->neuron_count; i++) {
        Neuron *n = g->neurons[i];
        if (n->is_refractory) continue;
        n->membrane_potential += (V_RESTING - n->membrane_potential) * 0.1 * g->dt;
        n->membrane_potential += n->external_current * g->dt;
        n->membrane_potential += ((double)rand() / RAND_MAX - 0.5) * 2.0 * NOISE_AMPLITUDE;
    }

    /* 4. Spike detection */
    for (size_t i = 0; i < g->neuron_count; i++) {
        Neuron *n = g->neurons[i];
        if (n->is_refractory) continue;
        if (n->membrane_potential >= n->threshold) {
            n->last_spike_time = g->current_time;
            n->is_refractory = true;
            n->membrane_potential = n->reset_potential;
            n->spike_trace += 1.0;
            neuron_record_spike(n, g->current_time);
            if (n->axon) {
                for (size_t j = 0; j < n->axon->out_count; j++) {
                    Synapse *s = n->axon->out_synapses[j];
                    s->last_activation = g->current_time;
                    synapse_schedule_spike(s, s->weight * SYNAPTIC_GAIN, g->step);
                }
            }
        }
    }

    /* 5. STDP */
    for (size_t i = 0; i < g->synapse_count; i++) {
        Synapse *s = g->synapses[i];
        if (!s->is_plastic) continue;
        double pre_spike  = s->presynaptic->last_spike_time;
        double post_spike = s->postsynaptic->last_spike_time;
        double delta = post_spike - pre_spike;
        if (fabs(delta) < STDP_TAU && delta != 0.0) {
            if (delta > 0)
                s->weight += s->learning_rate * exp(-delta / STDP_TAU);
            else
                s->weight -= s->learning_rate * exp(delta / STDP_TAU);
            if (s->weight > STDP_MAX_WEIGHT) s->weight = STDP_MAX_WEIGHT;
            if (s->weight < STDP_MIN_WEIGHT) s->weight = STDP_MIN_WEIGHT;
        }
    }

    g->step++;
    g->current_time += g->dt;
}

void graph_simulate(Graph *g, int steps) {
    printf("\n=== Simulation: %d steps, dt=%.2f ===\n", steps, g->dt);
    for (int i = 0; i < steps; i++) graph_step(g);
}

/* ============================================================
 *                    ASCII RASTER PLOT
 * ============================================================ */

void graph_raster(Graph *g, int width) {
    if (!g || width <= 0) return;
    printf("\n=== Raster plot ('#' = spike) ===\n");
    printf("     t=0");
    for (int i = 0; i < width - 8; i++) printf(" ");
    printf("t=%.0f\n", g->current_time);
    printf("     |");
    for (int i = 0; i < width; i++) printf("-");
    printf("|\n");

    for (size_t i = 0; i < g->neuron_count; i++) {
        Neuron *n = g->neurons[i];
        char tc = n->type == NEURON_EXCITATORY ? 'E' :
                  n->type == NEURON_INHIBITORY ? 'I' : 'M';
        printf("#%02zu%c |", n->id, tc);
        for (int x = 0; x < width; x++) {
            double t0 = (double)x / width * g->current_time;
            double t1 = (double)(x + 1) / width * g->current_time;
            bool hit = false;
            for (size_t k = 0; k < n->spike_count; k++) {
                if (n->spike_times[k] >= t0 && n->spike_times[k] < t1) {
                    hit = true; break;
                }
            }
            printf("%c", hit ? '#' : ' ');
        }
        printf("|\n");
    }
    printf("     |");
    for (int i = 0; i < width; i++) printf("-");
    printf("|\n");
    printf("\nLegend: E=excitatory, I=inhibitory, M=modulatory\n");
}

void graph_print_stats(Graph *g) {
    printf("\n=== Neuron statistics ===\n");
    printf("%-4s %-4s %-8s %-8s %-8s\n", "ID", "Type", "Spikes", "V (mV)", "I_ext");
    for (size_t i = 0; i < g->neuron_count; i++) {
        Neuron *n = g->neurons[i];
        const char *tn = n->type == NEURON_EXCITATORY ? "EXC" :
                         n->type == NEURON_INHIBITORY ? "INH" : "MOD";
        printf("%-4zu %-4s %-8zu %-8.1f %-8.1f\n",
               n->id, tn, n->spike_count, n->membrane_potential, n->external_current);
    }
}

void graph_print_weights(Graph *g) {
    printf("\n=== Synapse weights ===\n");
    for (size_t i = 0; i < g->synapse_count; i++) {
        Synapse *s = g->synapses[i];
        printf("  #%zu -> #%zu : w=%.4f, delay=%.1f\n",
               s->presynaptic_id, s->postsynaptic_id, s->weight, s->delay);
    }
}

/* ============================================================
 *                    NETWORK BUILDER
 * ============================================================ */

static double rand_range(double min, double range) {
    return min + ((double)rand() / RAND_MAX) * range;
}

void build_network(Graph *g, int n_exc, int n_inh, int n_mod) {
    int total = n_exc + n_inh + n_mod;
    Neuron **neurons = malloc((size_t)total * sizeof(Neuron *));
    if (!neurons) return;

    int idx = 0;

    for (int i = 0; i < n_exc; i++) {
        neurons[idx] = graph_add_neuron(g, NEURON_EXCITATORY);
        neuron_add_axon(neurons[idx]);
        neurons[idx]->external_current = rand_range(I_EXT_EXC_BASE, I_EXT_EXC_RANGE);
        idx++;
    }
    for (int i = 0; i < n_inh; i++) {
        neurons[idx] = graph_add_neuron(g, NEURON_INHIBITORY);
        neuron_add_axon(neurons[idx]);
        neurons[idx]->external_current = rand_range(I_EXT_INH_BASE, I_EXT_INH_RANGE);
        idx++;
    }
    for (int i = 0; i < n_mod; i++) {
        neurons[idx] = graph_add_neuron(g, NEURON_MODULATORY);
        neuron_add_axon(neurons[idx]);
        neurons[idx]->external_current = rand_range(I_EXT_MOD_BASE, I_EXT_MOD_RANGE);
        idx++;
    }

    int synapse_count = total * NET_SYNAPSES_PER_NODE;
    for (int i = 0; i < synapse_count; i++) {
        int pre_idx  = rand() % total;
        int post_idx = rand() % total;
        if (pre_idx == post_idx) continue;

        Neuron *pre  = neurons[pre_idx];
        Neuron *post = neurons[post_idx];

        SynapseType st;
        double weight;

        if (pre->type == NEURON_EXCITATORY) {
            st = SYNAPSE_EXCITATORY;
            weight = rand_range(W_EXC_MIN, W_EXC_RANGE);
        } else if (pre->type == NEURON_INHIBITORY) {
            st = SYNAPSE_INHIBITORY;
            weight = rand_range(W_INH_MIN, W_INH_RANGE);
        } else {
            st = SYNAPSE_MODULATORY;
            weight = rand_range(W_MOD_MIN, W_MOD_RANGE);
        }

        double delay = rand_range(SYNAPTIC_DELAY_MIN, SYNAPTIC_DELAY_MAX - SYNAPTIC_DELAY_MIN);

        graph_add_synapse(g, pre, post, st, weight, delay);
    }

    free(neurons);
}

/* ============================================================
 *                         MAIN
 * ============================================================ */
int main(void) {
    srand((unsigned)time(NULL));

    Graph *g = graph_create(SIM_DURATION, SIM_DT);
    if (!g) { fprintf(stderr, "Failed to create graph\n"); return 1; }

    build_network(g, NET_N_EXCITATORY, NET_N_INHIBITORY, NET_N_MODULATORY);

    printf("Network built: %zu neurons, %zu synapses\n",
           g->neuron_count, g->synapse_count);

    graph_simulate(g, SIM_STEPS);
    graph_print_stats(g);
    graph_raster(g, RASTER_WIDTH);

    graph_free(g);
    return 0;
}
