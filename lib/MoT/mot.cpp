#include "mot.h"

void init_mot_protocol() {
    init_physical_layer();
    init_mac_layer();
    init_network_layer();
    init_transport_layer();
    init_application_layer();
}