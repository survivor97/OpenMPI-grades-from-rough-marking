#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <vector>

enum CommunicationTag
{
    TAG_MASTER_SEND_ID,
    TAG_MASTER_SEND_DATA,
    TAG_MASTER_SEND_TERMINATE,
    TAG_SLAVE_SEND_RESULT,
};

int main(int argc, char* argv[])
{
    int world_size;
    int rank;  
    int init;
    
    const int NUMBER_OF_VALUES_PER_ROW = 6;      

    //--- INITIALIZE MPI --------------------------------------------------------------------------------

    init = MPI_Init(&argc, &argv);

    if (init != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating...\n");
        MPI_Abort(MPI_COMM_WORLD, init);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2) {
        printf("Not enough processes (minimun 2). Terminating...\n");
        MPI_Abort(MPI_COMM_WORLD, init);
    }

    //--- MASTER -----------------------------------------------------------------------------------------
    if (rank == 0) {
        FILE* dataFile;

        int totalSizeOfData = 0;

        fopen_s(&dataFile, "input/data_test.txt", "r");
        std::vector<double*> data;
        double *row_data = (double*)malloc(NUMBER_OF_VALUES_PER_ROW * sizeof(double));
        double myDouble;
        int counter = 0;

        //--- Scan data file ---
        assert(dataFile != NULL);

        while(fscanf_s(dataFile, "%lf", &myDouble) != EOF) {
            assert(row_data != NULL);
            row_data[counter] = myDouble;

            if (counter < NUMBER_OF_VALUES_PER_ROW -1) {      
                counter++;
            }

            else {
                data.push_back(row_data);
                counter = 0;   
                row_data = (double*)malloc(NUMBER_OF_VALUES_PER_ROW * sizeof(double));
            }            
        } 
        totalSizeOfData = data.size();
        fclose(dataFile);

        //----- Debug prints----
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < NUMBER_OF_VALUES_PER_ROW; j++) {
                printf("%lf ", data[i][j]);
            }
            printf("\n");
        }

        printf("Size of data: %d\n", totalSizeOfData);
        printf("Size of world: %d\n", world_size);

        //----- MPI-------------
        int nextIndexToSend = 0;

        //First send
        for (int i = 1; i < world_size; i++) {
            if (nextIndexToSend < data.size()) {
                MPI_Send(data[nextIndexToSend], NUMBER_OF_VALUES_PER_ROW, MPI_DOUBLE, i, TAG_MASTER_SEND_DATA, MPI_COMM_WORLD);
                MPI_Send(&nextIndexToSend, 1, MPI_INT, i, TAG_MASTER_SEND_ID, MPI_COMM_WORLD);
                nextIndexToSend++;
            }
        }  

        //Wait for response and send
        while (true) {
            MPI_Status status;
            double averageResult;
            MPI_Recv(&averageResult, 1, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD, &status);

            printf("average received from task %d: %lf\n", status.MPI_SOURCE, averageResult);

            //Break
            if (nextIndexToSend >= data.size())
            {
                break;
            }

            MPI_Send(data[nextIndexToSend], NUMBER_OF_VALUES_PER_ROW, MPI_DOUBLE, status.MPI_SOURCE, TAG_MASTER_SEND_DATA, MPI_COMM_WORLD);
            MPI_Send(&nextIndexToSend, 1, MPI_INT, status.MPI_SOURCE, TAG_MASTER_SEND_ID, MPI_COMM_WORLD);
            nextIndexToSend++;
        }

        // Terminate
        printf("Sending termination...\n");
        for (int i = 1; i < world_size; i++)
        {
            int response;
            MPI_Send(&response, 1, MPI_CHAR, i, TAG_MASTER_SEND_TERMINATE, MPI_COMM_WORLD);
        }
        //----------------------
    }

    //--- SLAVE -----------------------------------------------------------------------------------------
    else { 
        while (true) {            
            int sizeCountCheck;
            int id;
            int sizeCount = 0;
            double* data = nullptr;

            MPI_Status status;

            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);            

            // Terminate !
            if (status.MPI_TAG == TAG_MASTER_SEND_TERMINATE) break;

            printf("---------------------- START TASK\n");

            if (status.MPI_TAG == TAG_MASTER_SEND_DATA) {
                MPI_Get_count(&status, MPI_DOUBLE, &sizeCount);                
                data = new double[sizeCount];
                MPI_Recv(data, sizeCount, MPI_DOUBLE, 0, TAG_MASTER_SEND_DATA, MPI_COMM_WORLD, &status);
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
            if (status.MPI_TAG == TAG_MASTER_SEND_ID) {
                MPI_Recv(&id, sizeCount, MPI_INT, 0, TAG_MASTER_SEND_ID, MPI_COMM_WORLD, &status);
            }
            else {
                printf("Id not received correctly. Terminating...\n");
                MPI_Abort(MPI_COMM_WORLD, init);
            }

            printf("status count: %d\n", status.count); 

            printf("task %d received a chunk of size %d\n", rank, sizeCount);

            assert(data != nullptr);

            //Debug print
            for (int i = 0; i < NUMBER_OF_VALUES_PER_ROW; i++) {
                printf("element %d is: %lf\n", i, data[i]);
            }

            //Local calculations
            double sum = 0;
            double average = 0;                      

            for (int i = 0; i < NUMBER_OF_VALUES_PER_ROW; i++) {
                sum += sqrt(data[i]);
            }

            average = sum / NUMBER_OF_VALUES_PER_ROW;

            //Debug print
            printf("(id %d) Average from task %d: %lf\n", id, rank, average);

            //Return response
            MPI_Send(&average, 1, MPI_DOUBLE, 0, TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD);

            printf("---------------------- END TASK\n");
        }
    }
    //---------------------------------------------------------------------------------------------------

    MPI_Finalize();

}