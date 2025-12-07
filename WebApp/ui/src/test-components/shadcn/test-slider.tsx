import { Slider } from "@bluevoid/ui/slider";

function TestSlider() {
	return (
		<>
			<Slider defaultValue={[50]} max={100} step={1} className={"w-[60%]"} />
			<Slider defaultValue={[25, 75]} max={100} step={1} className={"w-[60%]"}/>
			<Slider defaultValue={[50]} max={100} step={10} className={"w-[60%]"} />
		</>
	);
}

export default TestSlider;
