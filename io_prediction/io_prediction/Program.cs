using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Reflection;
using System.Threading;
using FANNCSharp.Double;
using FANNCSharp;

namespace io_prediction
{
    class Program
    {
        static void Main(string[] args)
        {
            //config
            string WorkingPath = Properties.Settings.Default.PathToFile.Trim() != "" ? Properties.Settings.Default.PathToFile :
                Path.GetFullPath(System.IO.Path.GetDirectoryName(Assembly.GetEntryAssembly().Location));

            int NeuronInput = Properties.Settings.Default.NeuronInput;
            uint NeuronHidden = Properties.Settings.Default.NeuronHidden;
            int NeuronOutput = Properties.Settings.Default.NeuronOutput;

            //reading data from file 
            List<double> network_val = ReadFromFile();
            List<double> norm_data = normalizeList(network_val);

            int tr_quantity = (int)Math.Floor(norm_data.Count * Properties.Settings.Default.TrainingRatio);
            int verify_quantity = norm_data.Count - tr_quantity - NeuronInput - NeuronOutput;
            double[][] x_val = new double[tr_quantity][];
            double[][] y_val = new double[tr_quantity][];

            double[] norm_data_in = new double[NeuronInput];
            double[] norm_data_out = new double[NeuronOutput];

            for (int x = 0; x < tr_quantity; x++)
            {
                for (int j = 0; j < NeuronInput; j++)
                    norm_data_in[j] = norm_data[x + j];

                for (int k = 0; k < NeuronOutput; k++)
                    norm_data_out[k] = norm_data[x + NeuronInput + k];

                x_val[x] = norm_data_in;
                y_val[x] = norm_data_out;

            }

            double[][] x_val_test = new double[verify_quantity][];
            double[][] y_val_test = new double[verify_quantity][];

            for (int x = tr_quantity; x < norm_data.Count - NeuronInput - NeuronOutput; x++)
            {
                double[] norm_data_in_test = new double[NeuronInput];
                double[] norm_data_out_test = new double[NeuronOutput];

                for (int j = 0; j < NeuronInput; j++)
                    norm_data_in_test[j] = norm_data[x + j];

                for (int k = 0; k < NeuronOutput; k++)
                    norm_data_out_test[k] = norm_data[x + NeuronInput + k];

                x_val_test[x - tr_quantity] = norm_data_in_test;
                y_val_test[x - tr_quantity] = norm_data_out_test;

            }


            uint[] layers = { (uint)NeuronInput, NeuronHidden, (uint)NeuronOutput };
            NeuralNet net = new NeuralNet(FANNCSharp.NetworkType.SHORTCUT, layers);
            net.RandomizeWeights(0, 1);
            net.LearningRate = Properties.Settings.Default.LearningRate;
            TrainingData trdata = new TrainingData();
            trdata.SetTrainData(x_val, y_val);
            TrainingData trdata_test = new TrainingData();
            trdata_test.SetTrainData(x_val_test, y_val_test);
            Console.WriteLine("Start train");

            net.TrainingAlgorithm = (TrainingAlgorithm)Properties.Settings.Default.AlgorithType;

            string Filename = Properties.Settings.Default.Filename.Trim() != "" ? Properties.Settings.Default.Filename : "Default";

            net.ActivationFunctionHidden = ActivationFunction.SIGMOID_SYMMETRIC;
            net.ActivationFunctionOutput = ActivationFunction.SIGMOID_SYMMETRIC;
            net.SetScalingParams(trdata, 0, 1, 0, 1);
            net.SetScalingParams(trdata_test, 0, 1, 0, 1);
            net.TrainOnData(trdata, Properties.Settings.Default.MaxEpoch, 1, Properties.Settings.Default.DesiredError);
            Console.WriteLine("END");
            Console.ReadKey();

            double[] calc_out = new double[verify_quantity];
            double[] test_x = new double[NeuronInput];

            Console.WriteLine(net.MSE.ToString());
            net.Save(WorkingPath + "/Network_" + Filename + ".net");

            List<double> outputs = new List<double>();
            double[] predicted_y = new double[verify_quantity];

            StringBuilder sb = new StringBuilder();
            for (var iter = 0; iter < verify_quantity; iter++)
            {
                predicted_y = net.Run(x_val_test[iter]);
                sb.AppendLine(predicted_y[0] + ";" + y_val_test[iter][0]);
                x_val_test[iter].ToList().ForEach(x => Console.WriteLine(x));
                System.Console.WriteLine("Predicted output: " + predicted_y[0] +
                     "Expected = " + y_val_test[iter][0]
                     + "difference = "
                     + Math.Abs(predicted_y[0] - y_val_test[iter][0]));
                outputs.Add(predicted_y[0]);

            }
            File.WriteAllText(WorkingPath + "/output_" + Filename + ".data", sb.ToString());

            net.TestData(trdata_test);
            Console.WriteLine(net.MSE.ToString());
            Console.ReadLine();



        }

        private static List<double> ReadFromFile()
{
            List<double> network_val = new List<double>();
            using (TextReader reader = File.OpenText(Properties.Settings.Default.DataSource))
            {
                string line;
                string header = reader.ReadLine();
                while ((line = reader.ReadLine()) != null)
                {

                    var values = line.Split(',');
                    var date_in = values[1].Replace(".", ",");
                    network_val.Add(Convert.ToDouble(date_in));
                }
            }
return network_val;
}


        public static List<double> normalizeList(List<double> data)
        {

            double scaleMin = 0.0; //the normalized minimum desired
            double scaleMax = 1.0; //the normalized maximum desired
            double valueMax = data.Max();
            double valueMin = data.Min();
            double valueRange = valueMax - valueMin;
            double scaleRange = scaleMax - scaleMin;

            IEnumerable<double> normalized = data.Select(i =>
            ((scaleRange * (i - valueMin))
            / valueRange) + scaleMin);

            return normalized.ToList();

        }
    }
}


