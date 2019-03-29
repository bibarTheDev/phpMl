<?php

    namespace phpMl\classificacao;
    use Phpml\Classification\NaiveBayes;

    class analise {

        protected $classifier;

        public function __construct()
        {
            $this->classifier = new NaiveBayes();
            //NaiveBayes eh o tipo de algoritimo de classificacao
        }

        public function train($samples, $labels)
        {
            $this->classifier->train($samples, $labels);
        }

        public function predict($samples)
        {
            return $this->classifier->predict($samples);
        }
    }

?>
