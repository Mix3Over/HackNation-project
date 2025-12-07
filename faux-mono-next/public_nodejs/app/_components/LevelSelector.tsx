import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@bluevoid-test/ui/select";
import type { Level } from "../_hooks/useCacheDataTransform";

interface LevelSelectorProps {
  onValueChange: (value: Level) => void;
  value: Level;
  disabled?: boolean;
}
function LevelSelector(props: LevelSelectorProps) {
  const { onValueChange, value, disabled = false } = props;
  return (
    <Select disabled={disabled} onValueChange={onValueChange} value={value}>
      <SelectTrigger className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="sektor">Sektor</SelectItem>
        <SelectItem value="group">Group</SelectItem>
      </SelectContent>
    </Select>
  );
}

export default LevelSelector;
